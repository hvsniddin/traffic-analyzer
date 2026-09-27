import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'

import { SiteFooter } from '@/components/shared/site-footer'
import { SiteHeader } from '@/components/shared/site-header'
import { themeInitScript } from '@/components/shared/theme-toggle'

import './globals.css'

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] })
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] })

export const metadata: Metadata = {
	title: { default: 'Traffic Analyzer', template: '%s · Traffic Analyzer' },
	description:
		'Traffic event detection from a fixed road camera: live demo, annotated sample videos, EDA and technical report for the WIUT Hackathon 2026 CV track.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
	return (
		<html lang='en' suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable}`}>
			<head>
				<script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
			</head>
			<body className='flex min-h-dvh flex-col font-sans'>
				<SiteHeader />
				<main className='mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6'>{children}</main>
				<SiteFooter />
			</body>
		</html>
	)
}
