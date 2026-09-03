"use client";

import { type ChangeEvent, useEffect, useState } from "react";

import {
  apiBaseUrl,
  readJson,
  type DeleteDocumentResponse,
  type DocumentInfo,
  type MetricsResponse,
  type QueryResponse,
  type Source,
  type UploadResponse,
} from "@/lib/api";

function sourceFileUrl(source: Source) {
  return `${apiBaseUrl}/documents/${encodeURIComponent(source.source)}/file#page=${source.page}`;
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
    setUploadStatus("Extracting text and indexing document chunks...");
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      const response = await fetch(`${apiBaseUrl}/upload`, { method: "POST", body: formData });
      const data = await readJson<UploadResponse>(response);
      setDocuments(data.documents ?? []);
      setUploadStatus(`Indexed ${data.uploaded?.join(", ") ?? "PDFs"} (${data.total_chunks ?? 0} chunks).`);
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
                {uploading ? "Indexing..." : "Upload PDFs"}
              </button>
            </div>
            <p className="mt-2 text-sm text-slate-400" aria-live="polite">{uploadStatus}</p>
            {documents.length > 0 && (
              <ul className="mt-3 space-y-2 text-sm text-slate-300">
                {documents.map((document, index) => (
                  <li key={`${document.filename}-${index}`} className="flex items-center justify-between gap-3 rounded-lg bg-slate-950/60 px-3 py-2">
                    <span className="min-w-0 break-all">{document.filename}: {document.pages} pages, {document.chunks} chunks</span>
                    <button
                      type="button"
                      onClick={() => handleDelete(document.filename)}
                      disabled={deletingFilename !== null}
                      aria-label={`Remove ${document.filename}`}
                      title={`Remove ${document.filename}`}
                      className="shrink-0 rounded-md px-2 py-1 text-lg leading-none text-slate-400 transition hover:bg-rose-950/60 hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {deletingFilename === document.filename ? "…" : "×"}
                    </button>
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
              <p className="whitespace-pre-line leading-7 text-slate-200">{answer}</p>
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
                      <div className="mb-1 break-all text-xs uppercase tracking-[0.2em] text-cyan-400">{source.source}</div>
                      <div className="text-slate-300">Page {source.page}</div>
                      <div className="mt-2 text-slate-400">{source.text.slice(0, 280)}{source.text.length > 280 ? "..." : ""}</div>
                      {source.text.length > 280 && (
                        <details className="mt-2 text-slate-400">
                          <summary className="cursor-pointer text-cyan-300">View full passage</summary>
                          <p className="mt-2">{source.text}</p>
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
            <iframe
              src={sourceFileUrl(selectedSource)}
              title={`PDF viewer for ${selectedSource.source}, page ${selectedSource.page}`}
              className="min-h-0 flex-1 bg-white"
            />
          </section>
        </div>
      )}
    </main>
  );
}
