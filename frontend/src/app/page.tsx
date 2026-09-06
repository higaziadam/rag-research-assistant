"use client";

import { type ChangeEvent, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

import {
  apiBaseUrl,
  readJson,
  type DeleteDocumentResponse,
  type DocumentInfo,
  type Equation,
  type MetricsResponse,
  type QueryResponse,
  type Source,
  type UploadResponse,
} from "@/lib/api";

function sourceFileUrl(source: Source) {
  return `${apiBaseUrl}/documents/${encodeURIComponent(source.source)}/file#page=${source.page}`;
}

function sourceRegionPreviewUrl(source: Source) {
  if (!source.bounding_box || source.bounding_box.length !== 4) {
    return null;
  }
  const [x0, y0, x1, y1] = source.bounding_box;
  const parameters = new URLSearchParams({ page: String(source.page), x0: String(x0), y0: String(y0), x1: String(x1), y1: String(y1) });
  return `${apiBaseUrl}/documents/${encodeURIComponent(source.source)}/page-preview?${parameters}`;
}

function equationPreviewUrl(source: Source, equation: Equation) {
  if (equation.bounding_box.length !== 4) {
    return null;
  }
  const [x0, y0, x1, y1] = equation.bounding_box;
  const parameters = new URLSearchParams({ page: String(source.page), x0: String(x0), y0: String(y0), x1: String(x1), y1: String(y1) });
  return `${apiBaseUrl}/documents/${encodeURIComponent(source.source)}/page-preview?${parameters}`;
}

function sourceLabel(source: Source) {
  const type = source.type ?? "text";
  return type.charAt(0).toUpperCase() + type.slice(1);
}

function sourceContent(source: Source) {
  return source.type === "table" ? source.table || source.text : source.figure_caption || source.text;
}

function containsUnverifiedMath(content: string) {
  return /[∫∑√≤≥≈≠±×÷⎛⎝⎞⎠〈〉‖]/.test(content) || (content.split("\n").length >= 3 && /[=+*/^]/.test(content));
}

function documentContentSummary(document: DocumentInfo) {
  return Object.entries(document.content_counts ?? {})
    .filter(([, count]) => count > 0)
    .map(([type, count]) => `${count} ${type}${count === 1 ? "" : "s"}`)
    .join(", ");
}

function formatFileSize(fileSizeBytes?: number) {
  if (fileSizeBytes === undefined) {
    return "size unavailable";
  }
  if (fileSizeBytes < 1024) {
    return `${fileSizeBytes} B`;
  }

  const units = ["KB", "MB", "GB"];
  const unitIndex = Math.min(Math.floor(Math.log(fileSizeBytes) / Math.log(1024)) - 1, units.length - 1);
  const value = fileSizeBytes / 1024 ** (unitIndex + 1);
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function getOrCreateSessionId() {
  const storedSessionId = window.sessionStorage.getItem("rag-session-id");
  if (storedSessionId) {
    return storedSessionId;
  }

  const newSessionId = `browser-${crypto.randomUUID()}`;
  window.sessionStorage.setItem("rag-session-id", newSessionId);
  return newSessionId;
}

function FormattedContent({ content, className = "" }: { content: string; className?: string }) {
  return (
    <div className={className}>
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
          ul: ({ children }) => <ul className="mb-3 list-disc space-y-2 pl-5 last:mb-0">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 list-decimal space-y-2 pl-5 last:mb-0">{children}</ol>,
          code: ({ children }) => <code className="rounded bg-slate-800 px-1 py-0.5 font-mono text-sm">{children}</code>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function VerifiedMathContent({ latex, className = "" }: { latex: string; className?: string }) {
  return (
    <div className={`math-content ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{`$$\n${latex}\n$$`}</ReactMarkdown>
    </div>
  );
}

function EquationTranscription({ equation }: { equation: Equation }) {
  if (!equation.latex) {
    return <p className="text-xs text-amber-300">Equation detected. Open the cited PDF page to verify the original notation.</p>;
  }

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3">
      <p className="mb-2 text-xs font-medium text-amber-300">Local OCR transcription — verify against the cited PDF page</p>
      <VerifiedMathContent latex={equation.latex} className="overflow-x-auto text-slate-100" />
    </div>
  );
}

function AnswerMathEvidence({ sources }: { sources: Source[] }) {
  const equationSources = sources.flatMap((source) =>
    (source.equations ?? []).map((equation, index) => ({
      equation,
      key: `${source.source}-${source.page}-${equation.bounding_box.join("-")}-${index}`,
      previewUrl: equationPreviewUrl(source, equation),
      source,
    })),
  );
  const uniqueEquations = equationSources.filter((entry, index, entries) => entries.findIndex((candidate) => candidate.key === entry.key) === index).slice(0, 3);

  if (uniqueEquations.length === 0) {
    return null;
  }

  return (
    <section className="mt-5 border-t border-slate-800 pt-4" aria-label="Mathematical evidence">
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-amber-300">Original mathematical notation</p>
      <p className="mt-1 text-xs text-slate-400">Rendered directly from the cited PDF. Verify notation against the source page.</p>
      <div className="mt-3 grid gap-3">
        {uniqueEquations.map(({ equation, key, previewUrl, source }) => (
          <div key={key} className="rounded-lg border border-amber-500/30 bg-slate-900/60 p-3">
            {equation.latex ? (
              <VerifiedMathContent latex={equation.latex} className="overflow-x-auto text-slate-100" />
            ) : previewUrl ? (
              <a href={previewUrl} target="_blank" rel="noreferrer" className="block overflow-x-auto rounded bg-white p-2">
                {/* This is an authenticated/dynamic API crop, not a static Next.js image asset. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} alt={`Original equation from ${source.source}, page ${source.page}`} className="mx-auto max-h-44 max-w-full" />
              </a>
            ) : (
              <p className="text-sm text-amber-200">Equation detected. Open the cited PDF page to review it.</p>
            )}
            <p className="mt-2 text-xs text-slate-400">{source.source}, page {source.page}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Home() {
  const [query, setQuery] = useState("How does the reranker improve retrieval quality?");
  const [answer, setAnswer] = useState("Upload a PDF, then ask a document-grounded question.");
  const [sources, setSources] = useState<Source[]>([]);
  const [unsupported, setUnsupported] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [queryLatency, setQueryLatency] = useState<number | null>(null);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [uploadStatus, setUploadStatus] = useState("No PDFs indexed from this browser session.");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null);
  const [retryingFilename, setRetryingFilename] = useState<string | null>(null);
  const hasActiveIngestion = documents.some((document) => ["queued", "extracting", "embedding"].includes(document.status));

  useEffect(() => {
    getOrCreateSessionId();

    async function loadDashboardData() {
      const [metricsResult, documentsResult] = await Promise.allSettled([
        fetch(`${apiBaseUrl}/metrics`).then(readJson<MetricsResponse>),
        fetch(`${apiBaseUrl}/documents`).then(readJson<DocumentInfo[]>),
      ]);

      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value);
      }
      if (documentsResult.status === "fulfilled") {
        setDocuments(documentsResult.value);
      }
    }

    void loadDashboardData();
  }, []);

  useEffect(() => {
    if (!hasActiveIngestion) {
      return undefined;
    }

    const refreshDocuments = () => {
      void fetch(`${apiBaseUrl}/documents`)
        .then(readJson<DocumentInfo[]>)
        .then(setDocuments)
        .catch(() => undefined);
    };
    const documentPoller = window.setInterval(refreshDocuments, 2_000);
    return () => window.clearInterval(documentPoller);
  }, [hasActiveIngestion]);

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  async function handleUpload() {
    if (files.length === 0) {
      setUploadStatus("Choose at least one PDF before indexing.");
      return;
    }

    setUploading(true);
    setUploadStatus("Uploading PDFs and placing them in the indexing queue...");
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      const response = await fetch(`${apiBaseUrl}/upload`, { method: "POST", body: formData });
      const data = await readJson<UploadResponse>(response);
      setDocuments(data.documents ?? []);
      setUploadStatus(`Queued ${data.uploaded?.join(", ") ?? "PDFs"}. Progress updates below automatically.`);
      setFiles([]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown upload error.";
      setUploadStatus(`Upload failed: ${message}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk() {
    if (!query.trim()) {
      setAnswer("Enter a question before asking the research assistant.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({ query, top_k: 5, session_id: getOrCreateSessionId() }),
      });
      const data = await readJson<QueryResponse>(response);
      if (!data.answer) {
        throw new Error("The backend response did not include an answer.");
      }
      setAnswer(data.answer);
      setSources(data.sources ?? []);
      setUnsupported(data.unsupported ?? false);
      setQueryLatency(data.latency_ms ?? null);
      setSelectedSource(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown request error.";
      setAnswer(`Unable to answer the question: ${message}`);
      setSources([]);
      setUnsupported(true);
      setQueryLatency(null);
      setSelectedSource(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(filename: string) {
    if (!window.confirm(`Remove ${filename} from the assistant? This also deletes its saved PDF and index entries.`)) {
      return;
    }

    setDeletingFilename(filename);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await readJson<DeleteDocumentResponse>(response);
      setDocuments(data.documents);
      setSources((currentSources) => currentSources.filter((source) => source.source !== filename));
      if (selectedSource?.source === filename) {
        setSelectedSource(null);
      }
      setUploadStatus(`Removed ${data.deleted} from the assistant.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown deletion error.";
      setUploadStatus(`Could not remove ${filename}: ${message}`);
    } finally {
      setDeletingFilename(null);
    }
  }

  async function handleRetry(filename: string) {
    setRetryingFilename(filename);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${encodeURIComponent(filename)}/retry`, { method: "POST" });
      await readJson(response);
      setDocuments((currentDocuments) => currentDocuments.map((document) => (
        document.filename === filename
          ? { ...document, status: "queued", progress: 0, message: "Queued for retry.", error: undefined }
          : document
      )));
      setUploadStatus(`Retrying ${filename}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown retry error.";
      setUploadStatus(`Could not retry ${filename}: ${message}`);
    } finally {
      setRetryingFilename(null);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-10 flex items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-cyan-400">Multimodal RAG</p>
            <h1 className="mt-2 text-3xl font-bold">Research Assistant</h1>
          </div>
          <button
            type="button"
            className="rounded-full border border-cyan-500 bg-cyan-500/10 px-4 py-2 text-sm font-semibold text-cyan-300"
            onClick={() => window.open(`${apiBaseUrl}/docs`, "_blank")}
          >
            Open API Docs
          </button>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.4fr_0.9fr]">
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-2xl shadow-cyan-950/40">
            <label className="mb-2 block text-sm font-medium text-slate-300" htmlFor="pdf-upload">Documents</label>
            <div className="flex flex-col gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-950 p-4 sm:flex-row sm:items-center">
              <input
                id="pdf-upload"
                type="file"
                accept="application/pdf,.pdf"
                multiple
                onChange={handleFileSelection}
                className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-cyan-300 hover:file:bg-slate-700"
              />
              <button
                type="button"
                onClick={handleUpload}
                disabled={uploading || files.length === 0}
                className="shrink-0 rounded-xl border border-cyan-500 px-4 py-2 text-sm font-semibold text-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploading ? "Uploading..." : "Upload PDFs"}
              </button>
            </div>
            <p className="mt-2 text-sm text-slate-400" aria-live="polite">{uploadStatus}</p>
            {documents.length > 0 && (
              <ul className="mt-3 space-y-2 text-sm text-slate-300">
                {documents.map((document, index) => (
                  <li key={`${document.filename}-${index}`} className="flex items-center justify-between gap-3 rounded-lg bg-slate-950/60 px-3 py-2">
                    <div className="min-w-0 flex-1">
                      <div className="break-all">
                        {document.filename}: {formatFileSize(document.file_size_bytes)} · {document.pages} pages, {document.chunks} chunks
                      </div>
                      {document.status === "indexed" && documentContentSummary(document) && (
                        <p className="mt-1 text-xs text-slate-500">Extracted: {documentContentSummary(document)}</p>
                      )}
                      {document.status !== "indexed" && (
                        <div className="mt-2" aria-live="polite">
                          <div className="flex justify-between gap-3 text-xs text-slate-400">
                            <span>{document.status === "failed" ? document.error ?? document.message : document.message}</span>
                            <span>{document.progress}%</span>
                          </div>
                          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-800">
                            <div
                              className={`h-full transition-all ${document.status === "failed" ? "bg-rose-400" : "bg-cyan-400"}`}
                              style={{ width: `${document.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {document.status === "failed" && (
                        <button
                          type="button"
                          onClick={() => handleRetry(document.filename)}
                          disabled={retryingFilename !== null || deletingFilename !== null}
                          className="rounded-md px-2 py-1 text-xs font-medium text-cyan-300 transition hover:bg-cyan-950/50 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {retryingFilename === document.filename ? "Retrying..." : "Retry"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => handleDelete(document.filename)}
                        disabled={deletingFilename !== null || retryingFilename !== null}
                        aria-label={`Remove ${document.filename}`}
                        title={document.status === "queued" || document.status === "extracting" || document.status === "embedding" ? `Cancel and remove ${document.filename}` : `Remove ${document.filename}`}
                        className="rounded-md px-2 py-1 text-lg leading-none text-slate-400 transition hover:bg-rose-950/60 hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {deletingFilename === document.filename ? "…" : "×"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}

            <label className="mb-2 mt-6 block text-sm font-medium text-slate-300" htmlFor="question">Question</label>
            <textarea
              id="question"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="h-28 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-base text-slate-100 outline-none ring-0"
            />
            <p className="mt-2 text-xs text-slate-500">Math in answers and source passages supports LaTeX notation such as <code>$E = mc^2$</code> and <code>$$\int_0^1 x^2\,dx$$</code>.</p>
            <button
              type="button"
              onClick={handleAsk}
              className="mt-4 rounded-xl bg-cyan-500 px-4 py-2 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={loading}
            >
              {loading ? "Thinking..." : "Ask research assistant"}
            </button>

            <div className={`mt-8 rounded-xl border p-4 ${unsupported ? "border-amber-500/40 bg-amber-950/20" : "border-slate-800 bg-slate-950"}`} aria-live="polite">
              <p className="mb-2 text-xs uppercase tracking-[0.25em] text-slate-400">Answer</p>
              <FormattedContent content={answer} className="leading-7 text-slate-200" />
              <AnswerMathEvidence sources={sources} />
            </div>
          </section>

          <aside className="space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <p className="mb-4 text-xs uppercase tracking-[0.25em] text-slate-400">Evaluation</p>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between"><span>Recall@5</span><strong>{metrics ? metrics.recall_at_5.toFixed(2) : "-"}</strong></div>
                <div className="flex justify-between"><span>MRR</span><strong>{metrics ? metrics.mrr.toFixed(2) : "-"}</strong></div>
                <div className="flex justify-between"><span>Citation accuracy</span><strong>{metrics ? metrics.citation_accuracy.toFixed(2) : "-"}</strong></div>
                <div className="flex justify-between"><span>Faithfulness</span><strong>{metrics ? metrics.answer_faithfulness.toFixed(2) : "-"}</strong></div>
                <div className="flex justify-between"><span>Query latency</span><strong>{queryLatency !== null ? `${Math.round(queryLatency)} ms` : "-"}</strong></div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <p className="mb-4 text-xs uppercase tracking-[0.25em] text-slate-400">Sources</p>
              {sources.length === 0 ? (
                <p className="text-sm text-slate-400">No citations yet.</p>
              ) : (
                <ul className="space-y-3 text-sm text-slate-200">
                  {sources.map((source, index) => (
                    <li key={`${source.chunk_id}-${index}`} className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="break-all text-xs uppercase tracking-[0.2em] text-cyan-400">{source.source}</div>
                        <span className="rounded-full border border-cyan-800 bg-cyan-950/50 px-2 py-0.5 text-xs font-medium text-cyan-200">
                          {sourceLabel(source)}
                        </span>
                      </div>
                      <div className="mt-1 text-slate-300">Page {source.page}</div>
                      {source.type === "table" ? (
                        <pre className="mt-2 overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/60 p-3 font-mono text-xs leading-5 text-slate-300">
                          {sourceContent(source)}
                        </pre>
                      ) : containsUnverifiedMath(sourceContent(source)) ? (
                        <p className="mt-2 text-sm text-amber-200">
                          This passage contains unverified extracted notation. Preview the cited PDF page to review the original equation.
                        </p>
                      ) : (
                        <FormattedContent
                          content={`${sourceContent(source).slice(0, 280)}${sourceContent(source).length > 280 ? "..." : ""}`}
                          className="mt-2 text-slate-400"
                        />
                      )}
                      {sourceContent(source).length > 280 && (
                        <details className="mt-2 text-slate-400">
                          <summary className="cursor-pointer text-cyan-300">View full passage</summary>
                          {source.type === "table" ? (
                            <pre className="mt-2 overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/60 p-3 font-mono text-xs leading-5 text-slate-300">
                              {sourceContent(source)}
                            </pre>
                          ) : containsUnverifiedMath(sourceContent(source)) ? (
                            <p className="mt-2 text-sm text-amber-200">
                              Unverified mathematical notation is hidden here. Preview the cited PDF page to review the source.
                            </p>
                          ) : (
                            <FormattedContent content={sourceContent(source)} className="mt-2" />
                          )}
                        </details>
                      )}
                      {(source.quality_flags?.length ?? 0) > 0 && (
                        <p className="mt-3 text-xs text-amber-300">
                          Extraction note: {source.quality_flags?.join(", ").replaceAll("_", " ")}
                        </p>
                      )}
                      {(source.equations?.length ?? 0) > 0 && (
                        <details className="mt-3 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                          <summary className="cursor-pointer text-xs font-medium uppercase tracking-[0.16em] text-amber-300">
                            Math regions detected ({source.equations?.length})
                          </summary>
                          <div className="mt-3 space-y-3">
                            {source.equations?.map((equation, equationIndex) => (
                              <EquationTranscription key={`${source.chunk_id}-equation-${equationIndex}`} equation={equation} />
                            ))}
                          </div>
                        </details>
                      )}
                      <div className="mt-3 flex flex-wrap gap-3 text-xs font-medium">
                        <button
                          type="button"
                          onClick={() => setSelectedSource(source)}
                          className="rounded-md border border-cyan-700 px-3 py-2 text-cyan-300 transition hover:border-cyan-400 hover:text-cyan-100"
                        >
                          Preview cited page
                        </button>
                        <a
                          href={sourceFileUrl(source)}
                          target="_blank"
                          rel="noreferrer"
                          className="rounded-md border border-slate-700 px-3 py-2 text-slate-300 transition hover:border-slate-500 hover:text-white"
                        >
                          Open PDF
                        </a>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </aside>
        </div>
      </div>

      {selectedSource && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-labelledby="source-viewer-title"
        >
          <section className="flex h-[88vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl shadow-black/50">
            <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-5 py-4">
              <div className="min-w-0">
                <h2 id="source-viewer-title" className="truncate font-semibold text-white">
                  {selectedSource.source}
                </h2>
                <p className="mt-1 text-sm text-slate-400">Cited page {selectedSource.page}</p>
              </div>
              <div className="flex items-center gap-3">
                <a
                  href={sourceFileUrl(selectedSource)}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-md border border-cyan-700 px-3 py-2 text-sm font-medium text-cyan-300 transition hover:border-cyan-400 hover:text-cyan-100"
                >
                  Open in new tab
                </a>
                <button
                  type="button"
                  onClick={() => setSelectedSource(null)}
                  className="rounded-md border border-slate-700 px-3 py-2 text-sm font-medium text-slate-200 transition hover:border-slate-500 hover:text-white"
                >
                  Close
                </button>
              </div>
            </header>
            <div className="grid min-h-0 flex-1 lg:grid-cols-[0.8fr_1.2fr]">
              <section className="min-h-0 overflow-auto border-b border-slate-800 p-4 lg:border-b-0 lg:border-r">
                <p className="mb-3 text-xs uppercase tracking-[0.16em] text-slate-400">Cited source region</p>
                {sourceRegionPreviewUrl(selectedSource) ? (
                  // The image is rendered locally from the cited PDF coordinates.
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={sourceRegionPreviewUrl(selectedSource) ?? ""}
                    alt={`Extracted ${sourceLabel(selectedSource).toLowerCase()} region from page ${selectedSource.page}`}
                    className="w-full rounded-lg border border-slate-700 bg-white"
                  />
                ) : (
                  <p className="text-sm text-slate-400">No precise region was saved for this older source. Use the page viewer to verify it.</p>
                )}
                <p className="mt-3 text-sm text-slate-400">{sourceLabel(selectedSource)} evidence from page {selectedSource.page}.</p>
              </section>
              <iframe
                src={sourceFileUrl(selectedSource)}
                title={`PDF viewer for ${selectedSource.source}, page ${selectedSource.page}`}
                className="min-h-0 h-full w-full bg-white"
              />
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
