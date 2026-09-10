"use client";

import { type ChangeEvent, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

import {
  apiFetch,
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

const sourcePreviewCharacterLimit = 500;

function formatMetric(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(2) : "â€”";
}

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

function sourcePreviewContent(source: Source) {
  const content = sourceContent(source);
  return content.length > sourcePreviewCharacterLimit
    ? `${content.slice(0, sourcePreviewCharacterLimit)}...`
    : content;
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

function AnswerEvidencePreviews({
  sources,
  onPreview,
}: {
  sources: Source[];
  onPreview: (source: Source) => void;
}) {
  const evidenceEntries = sources.map((source) => {
    const equation = source.equations?.find((candidate) => candidate.bounding_box.length === 4);
    const previewUrl = equation ? equationPreviewUrl(source, equation) : sourceRegionPreviewUrl(source);
    const label = equation ? "Equation" : sourceLabel(source);
    return {
      key: `${source.source}-${source.page}-${source.type ?? "text"}-${equation?.bounding_box.join("-") ?? source.bounding_box?.join("-") ?? source.chunk_id}`,
      label,
      previewUrl,
      source,
    };
  });
  const uniqueEvidence = evidenceEntries
    .filter((entry, index, entries) => entries.findIndex((candidate) => candidate.key === entry.key) === index)
    .slice(0, 3);

  if (uniqueEvidence.length === 0) {
    return null;
  }

  return (
    <section className="mt-5 border-t border-slate-800 pt-4" aria-label="Supporting evidence">
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-300">Supporting evidence</p>
      <p className="mt-1 text-xs text-slate-400">Original regions rendered locally from the cited PDFs. Select a preview to inspect its full source page.</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {uniqueEvidence.map(({ key, label, previewUrl, source }) => (
          <article key={key} className="min-w-0 overflow-hidden rounded-lg border border-cyan-900/70 bg-slate-900/60 p-3">
            {previewUrl ? (
              <button
                type="button"
                onClick={() => onPreview(source)}
                className="block w-full overflow-hidden rounded border border-slate-700 bg-white text-left transition hover:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                aria-label={`Preview ${label.toLowerCase()} evidence from ${source.source}, page ${source.page}`}
              >
                {/* This is a dynamic crop from a local PDF API, not a static Next.js image asset. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt={`Cited ${label.toLowerCase()} evidence from ${source.source}, page ${source.page}`}
                  loading="lazy"
                  className="max-h-48 w-full object-contain"
                />
              </button>
            ) : (
              <div className="rounded border border-slate-700 bg-slate-950/70 p-3 text-xs text-slate-400">
                A precise crop is unavailable for this older indexed source. Open the cited page to verify the original evidence.
              </div>
            )}
            <div className="mt-2 flex items-center justify-between gap-2 text-xs">
              <span className="rounded-full border border-cyan-800 bg-cyan-950/50 px-2 py-0.5 font-medium text-cyan-200">{label}</span>
              <span className="shrink-0 text-slate-400">Page {source.page}</span>
            </div>
            <p className="mt-2 break-all text-xs text-slate-400">{source.source}</p>
            <button
              type="button"
              onClick={() => onPreview(source)}
              className="mt-2 text-xs font-medium text-cyan-300 transition hover:text-cyan-100"
            >
              Preview cited page
            </button>
          </article>
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
  const [browserDocumentNames, setBrowserDocumentNames] = useState<string[]>(() => {
    if (typeof window === "undefined") {
      return [];
    }
    const storedDocumentNames = window.sessionStorage.getItem("rag-document-names");
    if (!storedDocumentNames) {
      return [];
    }
    try {
      const parsedNames = JSON.parse(storedDocumentNames);
      return Array.isArray(parsedNames) && parsedNames.every((name) => typeof name === "string") ? parsedNames : [];
    } catch {
      window.sessionStorage.removeItem("rag-document-names");
      return [];
    }
  });
  const scopedDocuments = documents.filter((document) => browserDocumentNames.includes(document.filename));
  const hasActiveIngestion = (scopedDocuments.length > 0 ? scopedDocuments : documents)
    .some((document) => ["queued", "extracting", "embedding"].includes(document.status));
  const indexedBrowserDocumentNames = scopedDocuments
    .filter((document) => document.status === "indexed")
    .map((document) => document.filename);

  useEffect(() => {
    getOrCreateSessionId();
    const controller = new AbortController();

    async function loadDashboardData() {
      const [metricsResult, documentsResult] = await Promise.allSettled([
        apiFetch(`${apiBaseUrl}/metrics`, { signal: controller.signal }, 10_000).then(readJson<MetricsResponse>),
        apiFetch(`${apiBaseUrl}/documents`, { signal: controller.signal }, 10_000).then(readJson<DocumentInfo[]>),
      ]);

      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value);
      }
      if (documentsResult.status === "fulfilled") {
        setDocuments(documentsResult.value);
      }
    }

    void loadDashboardData();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!hasActiveIngestion) {
      return undefined;
    }

    const controller = new AbortController();
    let documentPoller: number | undefined;
    const refreshDocuments = async () => {
      try {
        const response = await apiFetch(`${apiBaseUrl}/documents`, { signal: controller.signal }, 10_000);
        setDocuments(await readJson<DocumentInfo[]>(response));
      } catch {
        // A later polling cycle retries transient backend failures.
      } finally {
        if (!controller.signal.aborted) {
          documentPoller = window.setTimeout(refreshDocuments, 2_000);
        }
      }
    };
    documentPoller = window.setTimeout(refreshDocuments, 2_000);
    return () => {
      controller.abort();
      if (documentPoller !== undefined) {
        window.clearTimeout(documentPoller);
      }
    };
  }, [hasActiveIngestion]);

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function rememberBrowserDocuments(filenames: string[]) {
    setBrowserDocumentNames((currentNames) => {
      const nextNames = [...new Set([...currentNames, ...filenames])];
      window.sessionStorage.setItem("rag-document-names", JSON.stringify(nextNames));
      return nextNames;
    });
  }

  function toggleDocumentScope(filename: string) {
    setBrowserDocumentNames((currentNames) => {
      const nextNames = currentNames.includes(filename)
        ? currentNames.filter((currentFilename) => currentFilename !== filename)
        : [...currentNames, filename];
      window.sessionStorage.setItem("rag-document-names", JSON.stringify(nextNames));
      return nextNames;
    });
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
      const response = await apiFetch(`${apiBaseUrl}/upload`, { method: "POST", body: formData }, 120_000);
      const data = await readJson<UploadResponse>(response);
      setDocuments(data.documents ?? []);
      rememberBrowserDocuments(data.uploaded ?? []);
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
    if (hasActiveIngestion) {
      setAnswer("Your PDF is still being indexed. Wait until its status is Indexed before asking a document-grounded question.");
      setSources([]);
      setUnsupported(false);
      setQueryLatency(null);
      return;
    }

    setLoading(true);
    try {
      const response = await apiFetch(`${apiBaseUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({
          query,
          top_k: 5,
          session_id: getOrCreateSessionId(),
          document_names: indexedBrowserDocumentNames,
        }),
      }, 60_000);
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
      const response = await apiFetch(`${apiBaseUrl}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await readJson<DeleteDocumentResponse>(response);
      setDocuments(data.documents);
      setBrowserDocumentNames((currentNames) => {
        const nextNames = currentNames.filter((currentFilename) => currentFilename !== filename);
        window.sessionStorage.setItem("rag-document-names", JSON.stringify(nextNames));
        return nextNames;
      });
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
      const response = await apiFetch(`${apiBaseUrl}/documents/${encodeURIComponent(filename)}/retry`, { method: "POST" });
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
            onClick={() => window.open(`${apiBaseUrl}/docs`, "_blank", "noopener,noreferrer")}
          >
            Open API Docs
          </button>
        </header>

        <div className="grid min-w-0 gap-6 lg:grid-cols-[1.4fr_0.9fr]">
          <section className="min-w-0 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-2xl shadow-cyan-950/40">
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
            {documents.length > 1 && (
              <p className="mt-2 text-xs text-slate-500">
                Search scope: {indexedBrowserDocumentNames.length > 0
                  ? `${indexedBrowserDocumentNames.length} selected document${indexedBrowserDocumentNames.length === 1 ? "" : "s"}`
                  : "all indexed documents"}. Select documents below to focus a query or comparison.
              </p>
            )}
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
                      {document.status === "indexed" && (
                        <label className="flex cursor-pointer items-center gap-1 rounded-md px-1 py-1 text-xs text-slate-400 hover:bg-slate-800">
                          <input
                            type="checkbox"
                            checked={browserDocumentNames.includes(document.filename)}
                            onChange={() => toggleDocumentScope(document.filename)}
                            className="accent-cyan-400"
                          />
                          Search
                        </label>
                      )}
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
              disabled={loading || hasActiveIngestion}
            >
              {loading ? "Thinking..." : hasActiveIngestion ? "Indexing PDFs..." : "Ask research assistant"}
            </button>
            {hasActiveIngestion && (
              <p className="mt-2 text-sm text-amber-300" role="status">
                Wait for indexing to finish before querying. This prevents incomplete documents from producing weak answers.
              </p>
            )}

            <div className={`mt-8 rounded-xl border p-4 ${unsupported ? "border-amber-500/40 bg-amber-950/20" : "border-slate-800 bg-slate-950"}`} aria-live="polite">
              <p className="mb-2 text-xs uppercase tracking-[0.25em] text-slate-400">Answer</p>
              <FormattedContent content={answer} className="leading-7 text-slate-200" />
              {!unsupported && <AnswerEvidencePreviews sources={sources} onPreview={setSelectedSource} />}
            </div>
          </section>

          <aside className="min-w-0 space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <p className="mb-4 text-xs uppercase tracking-[0.25em] text-slate-400">Evaluation</p>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between"><span>Recall@5</span><strong>{formatMetric(metrics?.recall_at_5)}</strong></div>
                <div className="flex justify-between"><span>MRR</span><strong>{formatMetric(metrics?.mrr)}</strong></div>
                <div className="flex justify-between"><span>Citation provenance</span><strong>{formatMetric(metrics?.citation_accuracy)}</strong></div>
                <div className="flex justify-between"><span>Faithfulness (review)</span><strong>{formatMetric(metrics?.answer_faithfulness)}</strong></div>
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
                    <li key={`${source.chunk_id}-${index}`} className="min-w-0 overflow-hidden rounded-xl border border-slate-800 bg-slate-950 p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="break-all text-xs uppercase tracking-[0.2em] text-cyan-400">{source.source}</div>
                        <span className="rounded-full border border-cyan-800 bg-cyan-950/50 px-2 py-0.5 text-xs font-medium text-cyan-200">
                          {sourceLabel(source)}
                        </span>
                      </div>
                      <div className="mt-1 text-slate-300">Page {source.page}</div>
                      {source.type === "table" ? (
                        <pre className="mt-2 max-w-full overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/60 p-3 font-mono text-xs leading-5 text-slate-300">
                          {sourcePreviewContent(source)}
                        </pre>
                      ) : containsUnverifiedMath(sourceContent(source)) ? (
                        <p className="mt-2 text-sm text-amber-200">
                          This passage contains unverified extracted notation. Preview the cited PDF page to review the original equation.
                        </p>
                      ) : (
                        <FormattedContent
                          content={sourcePreviewContent(source)}
                          className="mt-2 text-slate-400"
                        />
                      )}
                      {sourceContent(source).length > sourcePreviewCharacterLimit && (
                        <details className="mt-2 text-slate-400">
                          <summary className="cursor-pointer text-cyan-300">View full passage</summary>
                          {source.type === "table" ? (
                            <pre className="mt-2 max-h-80 max-w-full overflow-auto rounded-lg border border-slate-800 bg-slate-900/60 p-3 font-mono text-xs leading-5 text-slate-300">
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
