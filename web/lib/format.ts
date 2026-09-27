/** 83.4 -> "1:23.4" */
export function formatTimestamp(seconds: number, fractionDigits = 1) {
	if (!Number.isFinite(seconds) || seconds < 0) return '0:00'

	const minutes = Math.floor(seconds / 60)
	const rest = seconds - minutes * 60
	const whole = rest.toFixed(fractionDigits)
	const padded = rest < 10 ? `0${whole}` : whole

	return `${minutes}:${padded}`
}

export function formatSeconds(seconds: number) {
	if (!Number.isFinite(seconds)) return '—'
	return seconds < 60
		? `${seconds.toFixed(1)} s`
		: `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`
}

export function formatBytes(bytes: number) {
	if (!Number.isFinite(bytes)) return '—'
	if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`
	if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
	return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

export function humanizeKey(key: string) {
	const spaced = key.replace(/[_-]+/g, ' ').trim().toLowerCase()
	return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}
