import type { NextConfig } from 'next'

// Static export: the site is plain files on any static host.
// The live demo calls the inference API directly from the browser.
const nextConfig: NextConfig = {
	output: 'export',
	trailingSlash: true,
	images: { unoptimized: true },
}

export default nextConfig
