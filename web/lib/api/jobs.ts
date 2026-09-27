import { config, isMockApi } from '@/lib/config'
import type { ApiResult } from '@/types/api'
import type { TrafficEvent } from '@/types/events'
import {
	jobCreatedSchema,
	jobStateSchema,
	type JobCreated,
	type JobState,
} from '@/types/job'

import { apiRequest } from './api-request'

export function createJob(file: File, durationSec: number): Promise<ApiResult<JobCreated>> {
	if (isMockApi) return mockCreateJob(durationSec)

	const body = new FormData()
	body.append('file', file)

	return apiRequest({
		url: `${config.apiUrl}/api/jobs`,
		method: 'POST',
		body,
		schema: jobCreatedSchema,
		// Uploading a 200 MB file on a slow link takes a while.
		timeoutMs: 10 * 60_000,
	})
}

export function getJob(jobId: string): Promise<ApiResult<JobState>> {
	if (isMockApi) return mockGetJob(jobId)

	return apiRequest({
		url: `${config.apiUrl}/api/jobs/${encodeURIComponent(jobId)}`,
		schema: jobStateSchema,
		timeoutMs: 15_000,
	})
}

export const isJobActive = (state: JobState | null) =>
	state?.status === 'queued' || state?.status === 'running'

// ---------------------------------------------------------------------------
// Mock used only when NEXT_PUBLIC_API_URL is empty. The UI labels every mock
// response, so these fixed events are never presented as model output.

const MOCK_RUN_MS = 6000
const mockJobs = new Map<string, { startedAt: number; durationSec: number }>()

async function mockCreateJob(durationSec: number): Promise<ApiResult<JobCreated>> {
	const jobId = `mock-${crypto.randomUUID().slice(0, 8)}`
	mockJobs.set(jobId, { startedAt: Date.now(), durationSec })
	return { ok: true, status: 202, data: { job_id: jobId } }
}

function mockEvents(duration: number): TrafficEvent[] {
	const at = (ratio: number) => Math.round(duration * ratio * 10) / 10
	return [
		[at(0.1), at(0.22), 'jaywalking'],
		[at(0.18), at(0.26), 'failure_to_yield'],
		[at(0.4), at(0.47), 'illegal_turn'],
		[at(0.55), at(0.9), 'congestion'],
	]
}

async function mockGetJob(jobId: string): Promise<ApiResult<JobState>> {
	const job = mockJobs.get(jobId)

	if (!job) {
		return { ok: false, status: 404, message: 'Job not found' }
	}

	const elapsed = Date.now() - job.startedAt
	const base = { job_id: jobId, error: null }

	if (elapsed < 1000) {
		return {
			ok: true,
			status: 200,
			data: { ...base, status: 'queued', stage: 'queued', progress: null, result: null },
		}
	}

	if (elapsed < MOCK_RUN_MS) {
		return {
			ok: true,
			status: 200,
			data: {
				...base,
				status: 'running',
				stage: elapsed < 3500 ? 'detecting and tracking' : 'running event rules',
				progress: Math.min(0.99, (elapsed - 1000) / (MOCK_RUN_MS - 1000)),
				result: null,
			},
		}
	}

	return {
		ok: true,
		status: 200,
		data: {
			...base,
			status: 'done',
			stage: 'complete',
			progress: 1,
			result: { duration_sec: job.durationSec, events: mockEvents(job.durationSec), risk: null },
		},
	}
}
