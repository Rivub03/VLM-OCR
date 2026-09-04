import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document OCR",
  description: "Single-model printed document OCR",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

