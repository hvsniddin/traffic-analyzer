'use client'

import * as React from 'react'

import { createJob, getJob, isJobActive } from '@/lib/api/jobs'
import { config } from '@/lib/config'
import type { JobState } from '@/types/job'

import { useInterval } from './use-interval'

// A few failed polls in a row usually mean a cold start or a network blip.
const MAX_POLL_FAILURES = 4

export type AnalysisPhase = 'idle' | 'uploading' | 'polling' | 'done' | 'error'

export function useAnalysisJob() {
	const [phase, setPhase] = React.useState<AnalysisPhase>('idle')
	const [job, setJob] = React.useState<JobState | null>(null)
	const [error, setError] = React.useState<string | null>(null)
	const jobIdRef = React.useRef<string | null>(null)
	const failuresRef = React.useRef(0)
	const inFlightRef = React.useRef(false)
	// Increments on reset so late responses from an old job are ignored.
	const runRef = React.useRef(0)

	const poll = React.useCallback(async () => {
		const jobId = jobIdRef.current
		if (!jobId || inFlightRef.current) return

		const run = runRef.current
		inFlightRef.current = true
		const result = await getJob(jobId)
		inFlightRef.current = false

		if (run !== runRef.current) return

		if (!result.ok) {
			failuresRef.current += 1
			if (result.status === 404 || failuresRef.current >= MAX_POLL_FAILURES) {
				setError(result.message)
				setPhase('error')
			}
			return
		}

		failuresRef.current = 0
		setJob(result.data)

		if (result.data.status === 'done') setPhase('done')
		if (result.data.status === 'error') {
			setError(result.data.error ?? 'Inference failed without a message')
			setPhase('error')
		}
	}, [])

	useInterval(() => void poll(), phase === 'polling' ? config.pollIntervalMs : null)

	const start = React.useCallback(
		async (file: File, durationSec: number) => {
			const run = ++runRef.current
			jobIdRef.current = null
			failuresRef.current = 0
			setJob(null)
			setError(null)
			setPhase('uploading')

			const created = await createJob(file, durationSec)
			if (run !== runRef.current) return

			if (!created.ok) {
				setError(created.message)
				setPhase('error')
				return
			}

			jobIdRef.current = created.data.job_id
			setPhase('polling')
			void poll()
		},
		[poll],
	)

	const reset = React.useCallback(() => {
		runRef.current += 1
		jobIdRef.current = null
		setJob(null)
		setError(null)
		setPhase('idle')
	}, [])

	return {
		phase,
		job,
		error,
		isBusy: phase === 'uploading' || (phase === 'polling' && (job === null || isJobActive(job))),
		start,
		reset,
	}
}
