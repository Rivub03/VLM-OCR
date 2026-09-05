"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";

type Mode = "text" | "nid_front" | "nid_back" | "schema";
type Runtime = {
  model: string;
  profile: string;
  serving_engine: string;
  max_inference_concurrency: number;
  max_upload_mib: number;
  max_pdf_pages: number;
};
type PageResult = { page_number: number; text: string; markdown: string; fields?: Record<string, unknown>; warnings: string[] };
type OCRResult = { status: "completed"; result: PageResult[]; metadata: { request_id: string; model: string; serving_engine: string; page_count: number; elapsed_ms: number } };
type JobResponse = { job_id: string; status: "queued" | "running" | "completed" | "failed" | "cancelled"; result?: OCRResult; error?: string; detail?: string };

const NID_FRONT = JSON.stringify({
  name: "Full name in English", name_bn: "Full name in Bangla", father_name: "Father's name", mother_name: "Mother's name", dob: "Date of birth", nid_no: "NID number",
}, null, 2);
const NID_BACK = JSON.stringify({
  address_bn: "Address in Bangla", blood_group: "Blood group", place_of_birth: "Place of birth", issue_date: "Issue date", mrz_line1: "MRZ line 1", mrz_line2: "MRZ line 2", mrz_line3: "MRZ line 3",
}, null, 2);

function formatMs(value: number) {
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("text");
  const [schema, setSchema] = useState("");
  const [result, setResult] = useState<OCRResult | null>(null);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [error, setError] = useState("");
  const [processing, setProcessing] = useState(false);
  const [copied, setCopied] = useState(false);
  const activeJob = useRef<string | null>(null);
  const cancelled = useRef(false);
  const [tab, setTab] = useState<"text" | "markdown" | "json">("text");

  useEffect(() => {
    fetch("/api/runtime").then(async response => response.ok ? response.json() : null).then(setRuntime).catch(() => setRuntime(null));
  }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const acceptFile = (selected: File) => {
    if (!/^(image\/(jpeg|png|webp)|application\/pdf)$/.test(selected.type)) {
      setError("Use a PDF, JPEG, PNG, or WEBP file.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(selected.type === "application/pdf" ? null : URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };
  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const candidate = event.dataTransfer.files.item(0);
    if (candidate) acceptFile(candidate);
  };
  const select = (event: ChangeEvent<HTMLInputElement>) => {
    const candidate = event.target.files?.item(0);
    if (candidate) acceptFile(candidate);
  };
  const submit = async () => {
    if (!file || processing) return;
    if (mode === "schema") {
      try { if (!schema.trim() || typeof JSON.parse(schema) !== "object") throw new Error(); }
      catch { setError("Custom extraction needs a valid JSON object."); return; }
    }
    cancelled.current = false; setProcessing(true); setResult(null); setError("");
    const body = new FormData();
    body.set("file", file); body.set("mode", mode);
    if (mode === "schema") body.set("schema", schema);
    try {
      const response = await fetch("/api/jobs", { method: "POST", body });
      const payload = await response.json() as JobResponse;
      if (!response.ok) throw new Error(payload.detail || "OCR could not process the document.");
      activeJob.current = payload.job_id;
      if (cancelled.current) {
        await fetch(`/api/jobs/${encodeURIComponent(payload.job_id)}`, { method: "DELETE" });
        return;
      }
      while (!cancelled.current) {
        await new Promise(resolve => window.setTimeout(resolve, 750));
        const statusResponse = await fetch(`/api/jobs/${encodeURIComponent(payload.job_id)}`, { cache: "no-store" });
        const status = await statusResponse.json() as JobResponse;
        if (!statusResponse.ok) throw new Error(status.detail || "Unable to read OCR job status.");
        if (status.status === "completed" && status.result) { setResult(status.result); break; }
        if (status.status === "failed") throw new Error(status.error || "OCR could not process the document.");
        if (status.status === "cancelled") break;
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "OCR request failed.");
    } finally { activeJob.current = null; setProcessing(false); }
  };
  const cancel = async () => {
    cancelled.current = true;
    const jobId = activeJob.current;
    if (jobId) {
      try { await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" }); }
      catch { /* The interface can still return to idle if the cancel response is lost. */ }
    }
    activeJob.current = null;
    setProcessing(false);
    setError("OCR request cancelled.");
  };
  const combinedText = useMemo(() => result?.result.map(page => page.text).join("\n\n") ?? "", [result]);
  const copyText = async () => {
    try {
      let didCopy = false;
      if (navigator.clipboard?.writeText) {
        try { await navigator.clipboard.writeText(combinedText); didCopy = true; }
        catch { /* HTTP and restrictive browser policies often reject this API. */ }
      }
      if (!didCopy) {
        const input = document.createElement("textarea");
        try {
          input.value = combinedText;
          input.style.position = "fixed";
          input.style.opacity = "0";
          document.body.appendChild(input);
          input.focus(); input.select();
          if (!document.execCommand("copy")) throw new Error("Copy command was rejected.");
        } finally { input.remove(); }
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setError("Copy was blocked by the browser. Select the text and copy it manually.");
    }
  };
  const download = (extension: "txt" | "md" | "json") => {
    if (!result) return;
    const content = extension === "json" ? JSON.stringify(result, null, 2) : combinedText;
    const blob = new Blob([content], { type: extension === "json" ? "application/json" : "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob); const link = document.createElement("a");
    link.href = url; link.download = `ocr-${result.metadata.request_id}.${extension}`; link.click(); URL.revokeObjectURL(url);
  };
  const setPreset = (nextMode: Mode) => {
    setMode(nextMode);
    if (nextMode === "nid_front") setSchema(NID_FRONT);
    if (nextMode === "nid_back") setSchema(NID_BACK);
  };

  return (
    <main>
      <header className="topbar"><div className="brand"><span className="brand-mark">⌁</span><span>Document OCR</span></div><div className="runtime">{runtime ? <><strong>{runtime.profile}</strong><span>{runtime.serving_engine} · {runtime.max_inference_concurrency} concurrent</span></> : <span>Connecting to OCR service…</span>}</div></header>
      <section className="hero"><p className="eyebrow">SINGLE-MODEL DOCUMENT INTELLIGENCE</p><h1>Faithful OCR for printed documents.</h1><p>Upload a card, scan, or PDF. One OCR model handles every page; structured fields are validated locally without a judge model.</p></section>
      <section className="workspace">
        <div className="card input-card">
          <div className="section-heading"><div><p className="eyebrow">INPUT</p><h2>Document</h2></div>{file && <button className="quiet" onClick={() => { setFile(null); setResult(null); setError(""); }}>Remove</button>}</div>
          {!file ? <label className="dropzone" onDragOver={event => event.preventDefault()} onDrop={onDrop}><input type="file" accept="image/jpeg,image/png,image/webp,application/pdf" onChange={select} /><span className="upload-icon">↑</span><strong>Drop a document here</strong><span>or browse your device</span><small>PDF, JPEG, PNG, WEBP · {runtime?.max_upload_mib ?? 25} MiB max</small></label> : <div className="preview">{preview ? <Image src={preview} alt="Selected document" fill unoptimized sizes="(max-width: 800px) 94vw, 45vw" /> : <div className="pdf-preview"><span>PDF</span><strong>{file.name}</strong><small>Pages are rendered securely on the OCR server.</small></div>}{processing && <div className="scanline"><span>READING DOCUMENT</span></div>}<div className="file-meta"><strong>{file.name}</strong><span>{Math.ceil(file.size / 1024)} KB</span></div></div>}
          <fieldset><legend>Extraction mode</legend><div className="mode-grid"><button className={mode === "text" ? "active" : ""} onClick={() => setMode("text")}>Text & layout</button><button className={mode === "nid_front" ? "active" : ""} onClick={() => setPreset("nid_front")}>NID front</button><button className={mode === "nid_back" ? "active" : ""} onClick={() => setPreset("nid_back")}>NID back</button><button className={mode === "schema" ? "active" : ""} onClick={() => setMode("schema")}>Custom JSON</button></div></fieldset>
          {(mode === "schema" || mode.startsWith("nid_")) && <label className="schema-label">{mode === "schema" ? "Custom extraction schema" : "Preset NID fields"}<textarea value={schema} onChange={event => setSchema(event.target.value)} readOnly={mode.startsWith("nid_")} spellCheck={false} /></label>}
          {error && <p className="error">{error}</p>}
          {processing ? <button className="primary" onClick={cancel}>Cancel extraction<span>×</span></button> : <button className="primary" disabled={!file} onClick={submit}>Start extraction<span>→</span></button>}
        </div>
        <div className="card result-card">
          <div className="section-heading"><div><p className="eyebrow">OUTPUT</p><h2>{result ? "Extraction complete" : "Waiting for a document"}</h2></div>{result && <div className="downloads"><button onClick={() => download("txt")}>.txt</button><button onClick={() => download("md")}>.md</button><button onClick={() => download("json")}>.json</button></div>}</div>
          {!result ? <div className="empty"><span>⌁</span><strong>{processing ? "Reading your document" : "Results will appear here"}</strong><p>{processing ? "The server is processing each page once with bounded concurrency." : "Choose a document and begin extraction."}</p></div> : <><div className="tabs"><button className={tab === "text" ? "selected" : ""} onClick={() => setTab("text")}>Text</button><button className={tab === "markdown" ? "selected" : ""} onClick={() => setTab("markdown")}>Markdown</button><button className={tab === "json" ? "selected" : ""} onClick={() => setTab("json")}>JSON</button><button className="copy" onClick={copyText}>{copied ? "Copied" : "Copy"}</button></div><div className="result-body">{tab === "json" ? <pre>{JSON.stringify(result, null, 2)}</pre> : <>{result.result.map(page => <article className="page-result" key={page.page_number}><p className="page-label">PAGE {page.page_number}</p><pre>{tab === "text" ? page.text : page.markdown}</pre>{page.fields && <div className="fields"><p className="page-label">EXTRACTED FIELDS</p>{Object.entries(page.fields).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{String(value ?? "—")}</strong></div>)}</div>}{page.warnings.map(warning => <p className="warning" key={warning}>{warning}</p>)}</article>)}</>}</div><footer className="result-footer"><span>{result.metadata.model}</span><span>{result.metadata.page_count} page{result.metadata.page_count === 1 ? "" : "s"}</span><span>{formatMs(result.metadata.elapsed_ms)}</span></footer></>}
        </div>
      </section>
      <footer className="privacy">Uploaded source files are discarded after processing. Direct results remain in memory only for one hour and are cleared when the service restarts.</footer>
    </main>
  );
}
