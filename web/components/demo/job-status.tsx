import { CheckCircle2Icon, CircleDashedIcon, TriangleAlertIcon } from 'lucide-react'

import { Spinner } from '@/components/shared/spinner'
import { Badge } from '@/components/ui/badge'
import type { AnalysisPhase } from '@/hooks/use-analysis-job'
import { humanizeKey } from '@/lib/format'
import type { JobState } from '@/types/job'

const statusText = {
	queued: 'Queued',
	running: 'Running',
	done: 'Done',
	error: 'Failed',
} as const

type JobStatusProps = {
	phase: AnalysisPhase
	job: JobState | null
	elapsedSec: number
}

export function JobStatus({ phase, job, elapsedSec }: JobStatusProps) {
	const status = phase === 'uploading' ? null : job?.status ?? 'queued'
	const progress = job?.progress ?? null

	return (
		<div className='grid gap-3 rounded-2xl border bg-card p-5'>
			<div className='flex flex-wrap items-center justify-between gap-3'>
				<div className='flex items-center gap-2'>
					{status === 'done' ? (
						<Badge variant='success'>
							<CheckCircle2Icon /> Done
						</Badge>
					) : status === 'error' || phase === 'error' ? (
						<Badge variant='destructive'>
							<TriangleAlertIcon /> Failed
						</Badge>
					) : (
						<Badge variant='primary'>
							<Spinner className='size-3' /> {status ? statusText[status] : 'Uploading'}
						</Badge>
					)}
					<span className='text-sm text-muted-foreground'>
						{phase === 'uploading'
							? 'Sending the video to the server'
							: job?.stage
								? humanizeKey(job.stage)
								: 'Waiting for a worker'}
					</span>
				</div>
				<span className='font-mono text-xs text-muted-foreground tabular-nums'>
					{elapsedSec.toFixed(0)} s elapsed
					{job ? ` · job ${job.job_id}` : ''}
				</span>
			</div>

			{progress !== null ? (
				<div className='grid gap-1.5'>
					<div
						role='progressbar'
						aria-valuenow={Math.round(progress * 100)}
						aria-valuemin={0}
						aria-valuemax={100}
						className='h-2 overflow-hidden rounded-full bg-muted'
					>
						<div
							className='h-full rounded-full bg-primary transition-[width] duration-500'
							style={{ width: `${progress * 100}%` }}
						/>
					</div>
					<span className='font-mono text-xs text-muted-foreground tabular-nums'>
						{(progress * 100).toFixed(0)}% of frames processed
					</span>
				</div>
			) : status !== 'done' && phase !== 'error' ? (
				<div className='grid gap-1.5'>
					<div aria-hidden className='relative h-2 overflow-hidden rounded-full bg-muted'>
						<div className='absolute inset-y-0 w-1/3 animate-[slide_1.4s_ease-in-out_infinite] rounded-full bg-primary/60' />
					</div>
					<span className='flex items-center gap-1.5 text-xs text-muted-foreground'>
						<CircleDashedIcon className='size-3' />
						The server has not reported progress for this stage.
					</span>
				</div>
			) : null}
		</div>
	)
}
