'use client'

import * as React from 'react'

import { classMeta, classStyle, sortEvents } from '@/lib/events'
import { formatTimestamp } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { RiskPoint, TrafficEvent } from '@/types/events'
import type { Overlay } from '@/types/job'

import { DetectionOverlay, OverlayControls, type OverlayLayers } from './detection-overlay'
import { EventTimeline, type TimelineTrack } from './event-timeline'

type EventPlayerProps = {
	src: string | null
	poster?: string | null
	/** Used until the video reports its own duration. */
	duration: number
	tracks: TimelineTrack[]
	risk?: RiskPoint[] | null
	videoOverlay?: React.ReactNode
	/** Boxes, event flags, and scene geometry drawn in sync with playback. */
	overlay?: Overlay | null
}

export function EventPlayer({ src, poster, duration, tracks, risk, videoOverlay, overlay }: EventPlayerProps) {
	const videoRef = React.useRef<HTMLVideoElement>(null)
	const [currentTime, setCurrentTime] = React.useState(0)
	const [videoDuration, setVideoDuration] = React.useState<number | null>(null)
	const [listTrack, setListTrack] = React.useState(0)
	const [videoSize, setVideoSize] = React.useState<{ width: number; height: number } | null>(null)
	const [layers, setLayers] = React.useState<OverlayLayers>({ scene: true, boxes: true, flaggedOnly: false })

	// timeupdate fires ~4×/s; follow the frame clock while playing so the playhead is smooth.
	React.useEffect(() => {
		const video = videoRef.current
		if (!video) return

		let frame = 0
		const tick = () => {
			setCurrentTime(video.currentTime)
			if (!video.paused) frame = requestAnimationFrame(tick)
		}
		const onPlay = () => {
			cancelAnimationFrame(frame)
			frame = requestAnimationFrame(tick)
		}
		const onUpdate = () => setCurrentTime(video.currentTime)

		video.addEventListener('play', onPlay)
		video.addEventListener('seeked', onUpdate)
		video.addEventListener('timeupdate', onUpdate)
		return () => {
			cancelAnimationFrame(frame)
			video.removeEventListener('play', onPlay)
			video.removeEventListener('seeked', onUpdate)
			video.removeEventListener('timeupdate', onUpdate)
		}
	}, [src])

	const seek = React.useCallback((seconds: number) => {
		const video = videoRef.current
		setCurrentTime(seconds)
		if (!video) return
		video.currentTime = seconds
		void video.play().catch(() => {
			// Autoplay can be refused; the seek still applies.
		})
	}, [])

	const effectiveDuration = videoDuration ?? duration
	const shownTrack = tracks[Math.min(listTrack, tracks.length - 1)]

	return (
		<div className='grid gap-6'>
			<div className='relative overflow-hidden rounded-2xl border bg-black'>
				{src ? (
					<video
						ref={videoRef}
						src={src}
						poster={poster ?? undefined}
						controls
						playsInline
						muted
						preload='metadata'
						className='aspect-video w-full'
						onLoadedMetadata={(event) => {
							const value = event.currentTarget.duration
							if (Number.isFinite(value) && value > 0) setVideoDuration(value)
							const { videoWidth, videoHeight } = event.currentTarget
							if (videoWidth > 0 && videoHeight > 0) setVideoSize({ width: videoWidth, height: videoHeight })
						}}
					/>
				) : (
					<div className='flex aspect-video items-center justify-center text-sm text-white/70'>
						Video not available
					</div>
				)}
				{overlay && videoSize ? (
					<DetectionOverlay
						overlay={overlay}
						currentTime={currentTime}
						width={videoSize.width}
						height={videoSize.height}
						layers={layers}
						risk={risk}
					/>
				) : null}
				{videoOverlay}
			</div>

			{overlay ? <OverlayControls layers={layers} onChange={setLayers} /> : null}

			<EventTimeline
				tracks={tracks}
				duration={effectiveDuration}
				currentTime={currentTime}
				onSeek={seek}
				risk={risk}
			/>

			{shownTrack ? (
				<div className='grid gap-3'>
					{tracks.length > 1 ? (
						<div role='tablist' aria-label='Event list source' className='flex flex-wrap gap-1'>
							{tracks.map((track, index) => (
								<button
									key={track.name}
									role='tab'
									aria-selected={index === listTrack}
									onClick={() => setListTrack(index)}
									className={cn(
										'rounded-md px-3 py-1.5 text-sm',
										index === listTrack ? 'bg-muted font-medium' : 'text-muted-foreground hover:text-foreground',
									)}
								>
									{track.name} <span className='text-muted-foreground'>({track.events.length})</span>
								</button>
							))}
						</div>
					) : null}
					<EventList events={shownTrack.events} currentTime={currentTime} onSeek={seek} />
				</div>
			) : null}
		</div>
	)
}

function EventList({
	events,
	currentTime,
	onSeek,
}: {
	events: TrafficEvent[]
	currentTime: number
	onSeek: (seconds: number) => void
}) {
	if (events.length === 0) {
		return (
			<p className='rounded-xl border border-dashed p-4 text-sm text-muted-foreground'>
				No events in this video. An empty list is a valid answer: the task allows videos without events.
			</p>
		)
	}

	return (
		<div className='overflow-x-auto rounded-xl border'>
			<table className='w-full text-sm'>
				<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
					<tr>
						<th className='px-3 py-2 font-medium'>Event</th>
						<th className='px-3 py-2 font-medium'>Label id</th>
						<th className='px-3 py-2 text-right font-medium'>Start</th>
						<th className='px-3 py-2 text-right font-medium'>End</th>
						<th className='px-3 py-2 text-right font-medium'>Length</th>
					</tr>
				</thead>
				<tbody>
					{sortEvents(events).map((event) => {
						const [start, end, label] = event
						const active = currentTime >= start && currentTime < end
						return (
							<tr
								key={`${label}-${start}-${end}`}
								onClick={() => onSeek(start)}
								className={cn('cursor-pointer border-t hover:bg-muted/60', active && 'bg-primary/5')}
							>
								<td className='px-3 py-2'>
									<button
										type='button'
										onClick={(clickEvent) => {
											clickEvent.stopPropagation()
											onSeek(start)
										}}
										className='flex items-center gap-2 text-left font-medium outline-none focus-visible:underline'
									>
										<span className={cn('size-2 shrink-0 rounded-full', classStyle(label).dot)} />
										{classMeta[label].title}
									</button>
								</td>
								<td className='px-3 py-2 font-mono text-xs text-muted-foreground'>{label}</td>
								<td className='px-3 py-2 text-right font-mono tabular-nums'>{formatTimestamp(start)}</td>
								<td className='px-3 py-2 text-right font-mono tabular-nums'>{formatTimestamp(end)}</td>
								<td className='px-3 py-2 text-right font-mono text-muted-foreground tabular-nums'>
									{(end - start).toFixed(1)} s
								</td>
							</tr>
						)
					})}
				</tbody>
			</table>
		</div>
	)
}
