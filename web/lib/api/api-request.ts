import type { z } from 'zod'

import type { ApiResult } from '@/types/api'

const statusFallbacks: Record<number, string> = {
	400: 'The request was rejected',
	404: 'Not found',
	413: 'The file is larger than the server accepts',
	415: 'Only MP4 videos are accepted',
	422: 'The video could not be processed',
	429: 'The demo is busy, try again in a minute',
	500: 'The server failed to handle the request',
	502: 'The inference service is unavailable',
	503: 'The inference service is temporarily unavailable',
	504: 'The inference service timed out',
}

type ApiRequestArgs<TSchema extends z.ZodType> = {
	url: string
	schema: TSchema
	method?: 'GET' | 'POST'
	body?: FormData
	timeoutMs?: number
}

// FastAPI returns {"detail": "..."} or {"detail": [{msg}]}; plain {"error"} is also accepted.
function toMessage(data: unknown): string | null {
	if (typeof data === 'string') return data.trim() || null
	if (!data || typeof data !== 'object') return null

	const record = data as Record<string, unknown>
	const detail = record.detail ?? record.error ?? record.message

	if (typeof detail === 'string') return detail
	if (Array.isArray(detail)) {
		const messages = detail
			.map((item) => (item && typeof item === 'object' ? (item as { msg?: unknown }).msg : null))
			.filter((msg): msg is string => typeof msg === 'string')
		return messages.length ? messages.join('; ') : null
	}

	return null
}

/** Browser-side request that never throws for HTTP or network failures. */
export async function apiRequest<TSchema extends z.ZodType>({
	url,
	schema,
	method = 'GET',
	body,
	timeoutMs = 60_000,
}: ApiRequestArgs<TSchema>): Promise<ApiResult<z.infer<TSchema>>> {
	let response: Response

	try {
		response = await fetch(url, {
			method,
			body,
			cache: 'no-store',
			signal: AbortSignal.timeout(timeoutMs),
		})
	} catch (error) {
		const timedOut = error instanceof DOMException && error.name === 'TimeoutError'
		return {
			ok: false,
			status: 0,
			message: timedOut
				? 'The inference server did not answer in time'
				: 'Could not reach the inference server',
		}
	}

	const contentType = response.headers.get('content-type') ?? ''
	const data: unknown = contentType.includes('json')
		? await response.json().catch(() => null)
		: await response.text().catch(() => '')

	if (!response.ok) {
		return {
			ok: false,
			status: response.status,
			message: toMessage(data) ?? statusFallbacks[response.status] ?? 'Something went wrong',
		}
	}

	const parsed = schema.safeParse(data)

	if (!parsed.success) {
		return {
			ok: false,
			status: response.status,
			message: 'The server answered in an unexpected format',
		}
	}

	return { ok: true, status: response.status, data: parsed.data }
}
