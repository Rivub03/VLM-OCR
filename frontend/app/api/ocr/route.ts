import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const backend = process.env.BACKEND_URL;
  const key = process.env.OCR_API_KEY;
  if (!backend || !key) {
    return NextResponse.json({ detail: "The server-side OCR proxy is not configured." }, { status: 500 });
  }
  const form = await request.formData();
  const upstream = await fetch(`${backend}/api/v1/ocr`, {
    method: "POST",
    headers: { "X-API-Key": key },
    body: form,
    cache: "no-store",
  });
  const payload = await upstream.text();
  return new NextResponse(payload, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

