import type { Metadata } from "next";
import { AuthProvider } from "@/context/AuthContext";
import { InferenceProvider } from "@/context/InferenceContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "CV Pipeline — NPEC",
  description:
    "Plant organ segmentation and root tip detection for the NPEC research pipeline.",
};

/** Root layout wrapping all pages with the auth + inference providers and global styles. */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          <InferenceProvider>{children}</InferenceProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
