import "./globals.css";

import type { Metadata } from "next";
import localFont from "next/font/local";
import { Edu_AU_VIC_WA_NT_Pre } from 'next/font/google'
import { Toaster } from "@/components/ui/toaster";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

const edu = Edu_AU_VIC_WA_NT_Pre({
  subsets: ['latin'],
  display: 'swap',
  variable: "--font-edu",
  weight: "500",
})


export const metadata: Metadata = {
  title: "easy miner",
  description: "Send your Bybit orders to PTN",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${edu.variable} antialiased`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
