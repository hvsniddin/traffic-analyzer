import { promises as fs } from 'node:fs'
import path from 'node:path'

import type { Sample } from '@/types/sample'

// Written by scripts/build_sample_data.py; read at build time for the static export.
const SAMPLES_DIR = path.join(process.cwd(), 'public', 'samples')

export async function getSampleIds(): Promise<string[]> {
	const raw = await fs.readFile(path.join(SAMPLES_DIR, 'index.json'), 'utf8')
	return (JSON.parse(raw) as { samples: string[] }).samples
}

export async function getSample(id: string): Promise<Sample | null> {
	try {
		const raw = await fs.readFile(path.join(SAMPLES_DIR, `${id}.json`), 'utf8')
		return JSON.parse(raw) as Sample
	} catch {
		return null
	}
}

export async function getSamples(): Promise<Sample[]> {
	const ids = await getSampleIds()
	const samples = await Promise.all(ids.map(getSample))
	return samples.filter((sample): sample is Sample => sample !== null)
}

/** Public asset paths in the JSON are relative ("samples/C3896/preview.mp4"). */
export function assetUrl(relative: string | null) {
	return relative ? `/${relative.replace(/^\/+/, '')}` : null
}
