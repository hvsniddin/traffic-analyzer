'use client'

import * as React from 'react'

import { Pending } from '@/components/shared/pending'
import { Badge } from '@/components/ui/badge'
import { EventPlayer } from '@/components/video/event-player'
import type { TimelineTrack } from '@/components/video/event-timeline'
import type { RiskPoint } from '@/types/events'
import { overlaySchema, type Overlay } from '@/types/job'
import type { Sample } from '@/types/sample'

type SampleOverlay = { overlay: Overlay; risk: RiskPoint[] | null }

/** Boxes and risk exported by scripts/export_samples.py; absent until that runs. */
function useSampleOverlay(id: string) {
	const [data, setData] = React.useState<SampleOverlay | null>(null)
	React.useEffect(() => {
		let cancelled = false
		fetch(`/samples/${id}/overlay.json`)
			.then((response) => (response.ok ? response.json() : null))
			.then((json) => {
				if (cancelled || !json) return
				const parsed = overlaySchema.safeParse(json)
				if (!parsed.success) return
				const risk = Array.isArray(json.risk) ? (json.risk as RiskPoint[]) : null
				setData({ overlay: parsed.data, risk })
			})
			.catch(() => {
				// The player works without overlays.
			})
		return () => {
			cancelled = true
		}
	}, [id])
	return data
}

export function SampleResults({ sample }: { sample: Sample }) {
	const tracks: TimelineTrack[] = []
	if (sample.predictions) tracks.push({ name: 'Model predictions', events: sample.predictions.events })
	if (sample.labels) tracks.push({ name: 'Our dev labels', events: sample.labels.events })
	const annotation = useSampleOverlay(sample.id)

	const src = sample.annotated_url ?? sample.preview_url
	const annotated = Boolean(sample.annotated_url || annotation)

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
				risk={sample.predictions?.risk ?? annotation?.risk}
				overlay={annotation?.overlay}
				videoOverlay={
					<Badge className='absolute top-3 left-3 bg-black/70 text-white'>
						{annotated ? 'Annotated by our pipeline' : 'Original, 540p preview'}
					</Badge>
				}
			/>
			{annotation ? (
				<p className='text-xs text-muted-foreground'>
					Boxes, event flags, and the risk curve come from the same 2 FPS detector and tracker run as the model
					predictions. Red boxes are road users involved in an event.
				</p>
			) : null}
			{sample.labels?.note ? <p className='text-xs text-muted-foreground'>Dev labels: {sample.labels.note}</p> : null}
		</div>
	)
}
