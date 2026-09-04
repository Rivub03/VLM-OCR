import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const backend = process.env.BACKEND_URL;
  const key = process.env.OCR_API_KEY;
  if (!backend || !key) return NextResponse.json({ detail: "OCR proxy is not configured." }, { status: 500 });
  const upstream = await fetch(`${backend}/api/v1/runtime`, {
    headers: { "X-API-Key": key },
    cache: "no-store",
  });
  return NextResponse.json(await upstream.json(), { status: upstream.status });
}

