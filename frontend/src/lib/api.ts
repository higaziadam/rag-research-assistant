export const apiBaseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMilliseconds = 30_000,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
  const abortFromCaller = () => controller.abort();
  init.signal?.addEventListener("abort", abortFromCaller, { once: true });
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
    init.signal?.removeEventListener("abort", abortFromCaller);
  }
}

export type Equation = {
  latex: string;
  status: "needs_verification" | "source_only";
  confidence: number;
  bounding_box: number[];
};
export type Source = {
  chunk_id: string;
  source: string;
  page: number;
  text: string;
  table?: string;
  figure_caption?: string;
  // Older persisted API responses may not include this field.
  type?: "text" | "table" | "figure" | "equation";
  bounding_box?: number[];
  quality_flags?: string[];
  equations?: Equation[];
};
export type IngestionStatus = "queued" | "extracting" | "embedding" | "indexed" | "failed" | "cancelled";
export type DocumentInfo = {
  filename: string;
  pages: number;
  chunks: number;
  file_size_bytes?: number;
  status: IngestionStatus;
  progress: number;
  message: string;
  error?: string;
  job_id?: string;
  content_counts: Record<string, number>;
};
export type IngestionJob = {
  job_id: string;
  filename: string;
  status: IngestionStatus;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  error?: string;
};
export type QueryResponse = {
  answer?: string;
  answer_intent?: "definition" | "explanation" | "procedure" | "comparison" | "summary" | "visual";
  sources?: Source[];
  latency_ms?: number;
  unsupported?: boolean;
  detail?: string;
};
export type UploadResponse = { uploaded?: string[]; total_chunks?: number; documents?: DocumentInfo[]; jobs?: IngestionJob[]; detail?: string };
export type DeleteDocumentResponse = { deleted: string; documents: DocumentInfo[] };
export type MetricsResponse = {
  recall_at_5: number | null;
  mrr: number | null;
  citation_accuracy: number | null;
  answer_faithfulness: number | null;
  latency_ms: number | null;
};

export async function readJson<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T & { detail?: unknown };
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `The backend returned HTTP ${response.status}.`);
  }
  return data;
}
