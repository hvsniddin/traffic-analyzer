import { ArrowRightIcon } from 'lucide-react'
import Link from 'next/link'

import { Badge } from '@/components/ui/badge'
import { presentClasses, classMeta } from '@/lib/events'
import { formatSeconds } from '@/lib/format'
import type { Sample } from '@/types/sample'

export function SampleCard({ sample }: { sample: Sample }) {
	const events = sample.predictions?.events ?? sample.labels?.events ?? []
	const classes = presentClasses(events)

	return (
		<Link
			href={`/samples/${sample.id}/`}
			className='group grid overflow-hidden rounded-2xl border bg-card transition-colors hover:border-foreground/30'
		>
			{sample.poster_url ? (
				// eslint-disable-next-line @next/next/no-img-element -- static export
				<img src={`/${sample.poster_url}`} alt='' loading='lazy' className='aspect-video w-full object-cover' />
			) : (
				<div className='aspect-video bg-muted' />
			)}
			<div className='grid gap-3 p-4'>
				<div className='flex items-center justify-between gap-2'>
					<p className='font-mono font-semibold'>{sample.id}</p>
					<span className='text-xs text-muted-foreground'>{formatSeconds(sample.video.duration_sec)}</span>
				</div>
				<div className='flex flex-wrap gap-1.5'>
					{sample.predictions ? (
						<Badge variant='success'>{sample.predictions.events.length} predicted</Badge>
					) : (
						<Badge variant='outline'>Predictions pending</Badge>
					)}
					{sample.labels ? <Badge>{sample.labels.events.length} labelled</Badge> : null}
				</div>
				{classes.length ? (
					<p className='line-clamp-2 text-xs text-muted-foreground'>
						{classes.map((label) => classMeta[label].title).join(' · ')}
					</p>
				) : null}
				<span className='flex items-center gap-1 text-sm font-medium text-primary'>
					Open results <ArrowRightIcon className='size-4 transition-transform group-hover:translate-x-0.5' />
				</span>
			</div>
		</Link>
	)
}
