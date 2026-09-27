'use client'

import { DownloadIcon, FlaskConicalIcon, RotateCcwIcon } from 'lucide-react'
import * as React from 'react'

import { ErrorState, InlineError } from '@/components/shared/error-state'
import { Spinner } from '@/components/shared/spinner'
import { Button } from '@/components/ui/button'
import { EventPlayer } from '@/components/video/event-player'
import { useAnalysisJob } from '@/hooks/use-analysis-job'
import { useInterval } from '@/hooks/use-interval'
import { config, isMockApi } from '@/lib/config'
import { formatBytes, formatSeconds } from '@/lib/format'
import { checkVideoFile, type CheckedVideo } from '@/lib/video-file'

import { JobStatus } from './job-status'
import { UploadDropzone } from './upload-dropzone'

export function DemoConsole() {
	const { phase, job, error, isBusy, start, reset } = useAnalysisJob()
	const [video, setVideo] = React.useState<CheckedVideo | null>(null)
	const [fileError, setFileError] = React.useState<string | null>(null)
	const [isChecking, setIsChecking] = React.useState(false)
	const [startedAt, setStartedAt] = React.useState<number | null>(null)
	const [now, setNow] = React.useState(() => Date.now())

	useInterval(() => setNow(Date.now()), isBusy ? 500 : null)

	// Revoke the previous object URL whenever the selected video changes.
	React.useEffect(() => {
		if (!video) return
		return () => URL.revokeObjectURL(video.url)
	}, [video])

	async function handleFile(file: File) {
		setFileError(null)
		setIsChecking(true)
		const checked = await checkVideoFile(file)
		setIsChecking(false)

		if (!checked.ok) {
			setFileError(checked.message)
			return
		}

		reset()
		setVideo(checked.video)
	}

	function handleAnalyze() {
		if (!video || isBusy) return
		setStartedAt(Date.now())
		setNow(Date.now())
		void start(video.file, video.durationSec)
	}

	function handleClear() {
		reset()
		setVideo(null)
		setFileError(null)
		setStartedAt(null)
	}

	const result = phase === 'done' ? job?.result ?? null : null
	const elapsedSec = startedAt ? (now - startedAt) / 1000 : 0

	return (
		<div className='grid gap-6'>
			{isMockApi ? (
				<div className='flex items-start gap-3 rounded-xl border border-dashed p-4 text-sm'>
					<FlaskConicalIcon className='mt-0.5 size-4 shrink-0 text-muted-foreground' />
					<p>
						<span className='font-medium'>Mock mode.</span>{' '}
						<span className='text-muted-foreground'>
							This build has no inference API configured, so results below are fixed placeholder events, not model
							output. Set <code className='font-mono'>NEXT_PUBLIC_API_URL</code> to connect the real service.
						</span>
					</p>
				</div>
			) : null}

			{!video ? (
				<div className='grid gap-3'>
					<UploadDropzone disabled={isChecking} onFile={handleFile} />
					{isChecking ? (
						<p className='flex items-center gap-2 text-sm text-muted-foreground'>
							<Spinner /> Reading video metadata
						</p>
					) : null}
					{fileError ? <InlineError>{fileError}</InlineError> : null}
				</div>
			) : (
				<div className='flex flex-col gap-4 rounded-2xl border bg-card p-5 sm:flex-row sm:items-center sm:justify-between'>
					<div className='grid min-w-0 gap-0.5'>
						<p className='truncate font-medium'>{video.file.name}</p>
						<p className='font-mono text-xs text-muted-foreground tabular-nums'>
							{formatSeconds(video.durationSec)} · {video.width}×{video.height} · {formatBytes(video.file.size)}
						</p>
					</div>
					<div className='flex flex-wrap gap-2'>
						<Button variant='outline' onClick={handleClear} disabled={phase === 'uploading'}>
							<RotateCcwIcon /> Choose another
						</Button>
						{phase === 'idle' || phase === 'error' ? (
							<Button onClick={handleAnalyze}>{phase === 'error' ? 'Retry analysis' : 'Analyze video'}</Button>
						) : null}
						{result ? (
							<a
								className='inline-flex h-9 items-center gap-2 rounded-lg border bg-card px-4 text-sm font-medium hover:bg-muted'
								download={`${video.file.name.replace(/\.mp4$/i, '')}_events.json`}
								href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 1))}`}
							>
								<DownloadIcon className='size-4' /> Events JSON
							</a>
						) : null}
					</div>
				</div>
			)}

			{phase !== 'idle' ? <JobStatus phase={phase} job={job} elapsedSec={elapsedSec} /> : null}

			{phase === 'error' ? (
				<ErrorState
					className='rounded-2xl border'
					title='The analysis did not finish'
					description={error ?? 'Unknown error'}
					onRetry={video ? handleAnalyze : undefined}
				/>
			) : null}

			{video && result ? (
				<section className='grid gap-4'>
					<div className='flex flex-wrap items-baseline justify-between gap-2'>
						<h2 className='text-xl font-semibold tracking-tight'>
							{result.events.length} {result.events.length === 1 ? 'event' : 'events'} detected
						</h2>
						<p className='text-sm text-muted-foreground'>
							Click a segment or row to jump the video to its start. Red boxes are road users involved in an event.
							{result.risk ? '' : ' No risk curve: Part B is not enabled on this server.'}
						</p>
					</div>
					<EventPlayer
						src={result.annotated_video_url ?? video.url}
						duration={result.duration_sec}
						tracks={[{ name: isMockApi ? 'Mock events' : 'Detected events', events: result.events }]}
						risk={result.risk}
						overlay={result.overlay}
					/>
				</section>
			) : video && phase === 'idle' ? (
				<video src={video.url} controls muted playsInline className='aspect-video w-full rounded-2xl border bg-black' />
			) : null}

			<p className='text-xs text-muted-foreground'>
				Uploaded videos are used only to run this analysis and are deleted from the server after the job ends. Limits:
				MP4, {config.maxUploadMb} MB, {config.maxDurationSec} s. Inference runs on CPU, so expect a few minutes.
			</p>
		</div>
	)
}
