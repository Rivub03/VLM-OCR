import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

async function proxy(request: NextRequest, method: "GET" | "DELETE", context: { params: Promise<{ jobId: string }> }) {
  const backend = process.env.BACKEND_URL;
  const key = process.env.OCR_API_KEY;
  if (!backend || !key) {
    return NextResponse.json({ detail: "The server-side OCR proxy is not configured." }, { status: 500 });
  }
  const { jobId } = await context.params;
  const upstream = await fetch(`${backend}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    method,
    headers: { "X-API-Key": key },
    cache: "no-store",
    signal: request.signal,
  });
  return new NextResponse(await upstream.text(), {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export function GET(request: NextRequest, context: { params: Promise<{ jobId: string }> }) {
  return proxy(request, "GET", context);
}

export function DELETE(request: NextRequest, context: { params: Promise<{ jobId: string }> }) {
  return proxy(request, "DELETE", context);
}
