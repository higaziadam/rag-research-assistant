"use client";

import { type ChangeEvent, useEffect, useState } from "react";

import {
  apiBaseUrl,
  readJson,
  type DocumentInfo,
  type MetricsResponse,
  type QueryResponse,
  type Source,
  type UploadResponse,
} from "@/lib/api";

export default function Home() {
  const [query, setQuery] = useState("How does the reranker improve retrieval quality?");
  const [answer, setAnswer] = useState("Upload a PDF, then ask a document-grounded question.");
  const [sources, setSources] = useState<Source[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [queryLatency, setQueryLatency] = useState<number | null>(null);
  const [uploadStatus, setUploadStatus] = useState("No PDFs indexed from this browser session.");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const response = await fetch(`${apiBaseUrl}/metrics`);
        setMetrics(await readJson<MetricsResponse>(response));
      } catch {
        // Question and upload actions show connection errors when the backend is unavailable.
      }
    }
    void loadMetrics();
  }, []);

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
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
        body: JSON.stringify({ query, top_k: 5, session_id: "browser-session" }),
      });
      const data = await readJson<QueryResponse>(response);
      if (!data.answer) {
        throw new Error("The backend response did not include an answer.");
      }
      setAnswer(data.answer);
      setSources(data.sources ?? []);
      setQueryLatency(data.latency_ms ?? null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown request error.";
      setAnswer(`Unable to answer the question: ${message}`);
      setSources([]);
      setQueryLatency(null);
    } finally {
      setLoading(false);
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
              <ul className="mt-3 space-y-1 text-sm text-slate-300">
                {documents.map((document) => (
                  <li key={document.filename}>{document.filename}: {document.pages} pages, {document.chunks} chunks</li>
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

            <div className="mt-8 rounded-xl border border-slate-800 bg-slate-950 p-4" aria-live="polite">
              <p className="mb-2 text-xs uppercase tracking-[0.25em] text-slate-400">Answer</p>
              <p className="leading-7 text-slate-200">{answer}</p>
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
                  {sources.map((source) => (
                    <li key={source.chunk_id} className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                      <div className="mb-1 text-xs uppercase tracking-[0.2em] text-cyan-400">{source.source}</div>
                      <div className="text-slate-300">Page {source.page}</div>
                      <div className="mt-2 text-slate-400">{source.text}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
