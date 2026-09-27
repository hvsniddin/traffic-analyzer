'use client'

import { Pending } from '@/components/shared/pending'
import { Badge } from '@/components/ui/badge'
import { EventPlayer } from '@/components/video/event-player'
import type { TimelineTrack } from '@/components/video/event-timeline'
import type { Sample } from '@/types/sample'

export function SampleResults({ sample }: { sample: Sample }) {
	const tracks: TimelineTrack[] = []
	if (sample.predictions) tracks.push({ name: 'Model predictions', events: sample.predictions.events })
	if (sample.labels) tracks.push({ name: 'Our dev labels', events: sample.labels.events })

	const src = sample.annotated_url ?? sample.preview_url

	return (
		<div className='grid gap-4'>
			{!sample.predictions ? (
				<Pending title='Model predictions for this video are not exported yet'>
					The timeline below shows {sample.labels ? 'our hand-made dev labels only' : 'no events yet'}. It fills in from{' '}
					<code className='font-mono'>predictions_samples.json</code> when the pipeline run completes.
				</Pending>
			) : null}
			<EventPlayer
				src={src ? `/${src}` : null}
				poster={sample.poster_url ? `/${sample.poster_url}` : null}
				duration={sample.video.duration_sec}
				tracks={tracks.length ? tracks : [{ name: 'Events', events: [] }]}
				risk={sample.predictions?.risk}
				videoOverlay={
					<Badge className='absolute top-3 left-3 bg-black/70 text-white'>
						{sample.annotated_url ? 'Annotated by our pipeline' : 'Original, 540p preview'}
					</Badge>
				}
			/>
			{sample.labels?.note ? <p className='text-xs text-muted-foreground'>Dev labels: {sample.labels.note}</p> : null}
		</div>
	)
}
