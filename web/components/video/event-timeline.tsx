'use client'

import * as React from 'react'

import { classMeta, classStyle, presentClasses } from '@/lib/events'
import { formatTimestamp } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { RiskPoint, TrafficEvent } from '@/types/events'

export type TimelineTrack = {
	name: string
	events: TrafficEvent[]
}

type EventTimelineProps = {
	tracks: TimelineTrack[]
	duration: number
	currentTime: number
	onSeek: (seconds: number) => void
	risk?: RiskPoint[] | null
}

const LABEL_COL = 'grid grid-cols-[7.5rem_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[11rem_minmax(0,1fr)]'

function pct(value: number, duration: number) {
	return `${Math.min(100, Math.max(0, (value / duration) * 100))}%`
}

function tickStep(duration: number) {
	const candidates = [5, 10, 15, 30, 60, 120, 300]
	return candidates.find((step) => duration / step <= 8) ?? 600
}

export function EventTimeline({ tracks, duration, currentTime, onSeek, risk }: EventTimelineProps) {
	const [hovered, setHovered] = React.useState<{ event: TrafficEvent; x: number; y: number } | null>(null)
	const containerRef = React.useRef<HTMLDivElement>(null)
	const safeDuration = duration > 0 ? duration : 1
	const step = tickStep(safeDuration)
	const ticks = Array.from({ length: Math.floor(safeDuration / step) + 1 }, (_, i) => i * step)

	function seekFromPointer(event: React.MouseEvent<HTMLDivElement>) {
		const rect = event.currentTarget.getBoundingClientRect()
		const ratio = (event.clientX - rect.left) / rect.width
		onSeek(Math.min(safeDuration, Math.max(0, ratio * safeDuration)))
	}

	function showTooltip(event: TrafficEvent, target: HTMLElement) {
		const box = containerRef.current?.getBoundingClientRect()
		const rect = target.getBoundingClientRect()
		if (!box) return
		setHovered({ event, x: rect.left - box.left + rect.width / 2, y: rect.top - box.top })
	}

	const playhead = (
		<div
			aria-hidden
			className='pointer-events-none absolute inset-y-0 w-px bg-foreground'
			style={{ left: pct(currentTime, safeDuration) }}
		/>
	)

	return (
		<div ref={containerRef} className='relative grid gap-4' onMouseLeave={() => setHovered(null)}>
			{tracks.map((track) => {
				const classes = presentClasses(track.events)

				return (
					<div key={track.name} className='grid gap-1.5'>
						{tracks.length > 1 ? (
							<p className='text-xs font-medium tracking-wide text-muted-foreground uppercase'>
								{track.name}
							</p>
						) : null}

						{classes.length === 0 ? (
							<div className={LABEL_COL}>
								<span className='text-xs text-muted-foreground'>No events</span>
								<div
									className='relative h-7 cursor-pointer rounded-md bg-muted'
									onClick={seekFromPointer}
								>
									{playhead}
								</div>
							</div>
						) : null}

						{classes.map((label) => (
							<div key={label} className={LABEL_COL}>
								<span className='flex min-w-0 items-center gap-2 text-xs'>
									<span className={cn('size-2 shrink-0 rounded-full', classStyle(label).dot)} />
									<span className='truncate' title={classMeta[label].title}>
										{classMeta[label].title}
									</span>
								</span>
								<div
									className='relative h-7 cursor-pointer rounded-md bg-muted'
									onClick={seekFromPointer}
									title='Click to seek'
								>
									{track.events
										.filter((event) => event[2] === label)
										.map((event) => {
											const [start, end] = event
											const active = currentTime >= start && currentTime < end
											return (
												<button
													key={`${start}-${end}`}
													type='button'
													aria-label={`${classMeta[label].title}, ${formatTimestamp(start)} to ${formatTimestamp(end)}. Seek to start.`}
													onClick={(clickEvent) => {
														clickEvent.stopPropagation()
														onSeek(start)
													}}
													onMouseEnter={(mouseEvent) => showTooltip(event, mouseEvent.currentTarget)}
													onFocus={(focusEvent) => showTooltip(event, focusEvent.currentTarget)}
													onBlur={() => setHovered(null)}
													className={cn(
														'absolute inset-y-1 min-w-1.5 rounded-sm ring-2 ring-muted outline-none transition-opacity focus-visible:ring-foreground',
														classStyle(label).bar,
														active ? 'opacity-100 ring-foreground/60' : 'opacity-80 hover:opacity-100',
													)}
													style={{
														left: pct(start, safeDuration),
														width: pct(end - start, safeDuration),
													}}
												/>
											)
										})}
									{playhead}
								</div>
							</div>
						))}
					</div>
				)
			})}

			{risk && risk.length > 0 ? (
				<div className={LABEL_COL}>
					<span className='text-xs'>Accident risk</span>
					<RiskStrip risk={risk} duration={safeDuration} currentTime={currentTime} onSeek={onSeek} />
				</div>
			) : null}

			<div className={LABEL_COL}>
				<span />
				<div className='relative h-4 font-mono text-[10px] text-muted-foreground tabular-nums'>
					{ticks.map((tick) => (
						<span
							key={tick}
							className='absolute -translate-x-1/2 first:translate-x-0'
							style={{ left: pct(tick, safeDuration) }}
						>
							{formatTimestamp(tick, 0)}
						</span>
					))}
				</div>
			</div>

			{hovered ? (
				<div
					role='tooltip'
					className='pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-lg border bg-card px-3 py-2 text-xs shadow-lg'
					style={{ left: hovered.x, top: hovered.y - 6 }}
				>
					<p className='font-medium'>{classMeta[hovered.event[2]].title}</p>
					<p className='font-mono text-muted-foreground tabular-nums'>
						{formatTimestamp(hovered.event[0])} – {formatTimestamp(hovered.event[1])} ·{' '}
						{(hovered.event[1] - hovered.event[0]).toFixed(1)} s
					</p>
				</div>
			) : null}
		</div>
	)
}

/** Risk curve on the same time axis as the lanes, with the θ = 0.5 alarm line. */
function RiskStrip({
	risk,
	duration,
	currentTime,
	onSeek,
}: {
	risk: RiskPoint[]
	duration: number
	currentTime: number
	onSeek: (seconds: number) => void
}) {
	// Downsample to at most ~600 points so long clips stay light.
	const stride = Math.max(1, Math.ceil(risk.length / 600))
	const points = risk.filter((_, index) => index % stride === 0)
	const path = points
		.map(([t, score], index) => `${index ? 'L' : 'M'}${(t / duration) * 1000},${(1 - score) * 100}`)
		.join(' ')

	return (
		<div
			className='relative h-14 cursor-pointer rounded-md bg-muted'
			onClick={(event) => {
				const rect = event.currentTarget.getBoundingClientRect()
				onSeek(((event.clientX - rect.left) / rect.width) * duration)
			}}
		>
			<svg viewBox='0 0 1000 100' preserveAspectRatio='none' className='absolute inset-0 size-full'>
				<line x1='0' x2='1000' y1='50' y2='50' stroke='var(--grid)' strokeWidth='1' vectorEffect='non-scaling-stroke' />
				<path
					d={`${path} L1000,100 L0,100 Z`}
					fill='var(--family-collision)'
					fillOpacity='0.15'
				/>
				<path
					d={path}
					fill='none'
					stroke='var(--family-collision)'
					strokeWidth='2'
					vectorEffect='non-scaling-stroke'
				/>
			</svg>
			<span className='absolute end-1 top-[calc(50%-14px)] font-mono text-[10px] text-muted-foreground'>θ 0.5</span>
			<div
				aria-hidden
				className='pointer-events-none absolute inset-y-0 w-px bg-foreground'
				style={{ left: pct(currentTime, duration) }}
			/>
		</div>
	)
}
