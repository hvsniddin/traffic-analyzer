function toNumber(value: string | undefined, fallback: number) {
	const parsed = Number(value)
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export const config = {
	/** Empty string means the demo runs against the labelled mock. */
	apiUrl: (process.env.NEXT_PUBLIC_API_URL ?? '').replace(/\/+$/, ''),
	maxUploadMb: toNumber(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB, 200),
	maxDurationSec: toNumber(process.env.NEXT_PUBLIC_MAX_DURATION_SEC, 120),
	repoUrl:
		process.env.NEXT_PUBLIC_REPO_URL ??
		'https://github.com/hvsniddin/traffic-analyzer',
	pollIntervalMs: 1500,
} as const

export const isMockApi = config.apiUrl === ''
