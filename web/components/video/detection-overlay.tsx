'use client'

import * as React from 'react'

import { classMeta } from '@/lib/events'
import { cn } from '@/lib/utils'
import type { EventClass, RiskPoint } from '@/types/events'
import type { Overlay } from '@/types/job'

// Detector class ids from weights/best.pt (detections/classes.py).
const DETECTOR_CLASSES = ['bicycle', 'bus', 'car', 'green light', 'motorcycle', 'person', 'red light', 'truck']
const PERSON = 5
const LIGHTS = new Set([3, 6])

const SCENE_COLORS: Record<string, string> = {
	lane: '#38bdf8',
	crosswalk: '#facc15',
	intersection: '#818cf8',
	island: '#f87171',
	waiting_zone: '#2dd4bf',
	stop_line: '#ef4444',
	solid_line: '#fb923c',
}

export type OverlayLayers = { visible: boolean; scene: boolean; boxes: boolean; flaggedOnly: boolean }

/** Index of the last element with t <= time, or -1. */
function lastAtOrBefore<T>(items: T[], time: number, at: (item: T) => number) {
	let low = 0
	let high = items.length - 1
	let found = -1
	while (low <= high) {
		const mid = (low + high) >> 1
		if (at(items[mid]) <= time) {
			found = mid
			low = mid + 1
		} else high = mid - 1
	}
	return found
}

type DetectionOverlayProps = {
	overlay: Overlay
	currentTime: number
	width: number
	height: number
	layers: OverlayLayers
	risk?: RiskPoint[] | null
}

export function DetectionOverlay({ overlay, currentTime, width, height, layers, risk }: DetectionOverlayProps) {
	if (!layers.visible) return null
	const frames = overlay.frames
	const index = lastAtOrBefore(frames, currentTime, (frame) => frame.t)
	// Hold a sample until the next one; hide boxes if the gap is unusually long.
	const period = frames.length > 1 ? (frames[frames.length - 1].t - frames[0].t) / (frames.length - 1) : 1
	const frame = index >= 0 && currentTime - frames[index].t <= 2 * period ? frames[index] : null
	const riskIndex = risk ? lastAtOrBefore(risk, currentTime, (point) => point[0]) : -1
	const riskNow = risk && riskIndex >= 0 ? risk[riskIndex][1] : null
	const stroke = Math.max(2, width / 640)
	const font = Math.max(12, width / 90)

	return (
		<>
			<svg
				viewBox={`0 0 ${width} ${height}`}
				preserveAspectRatio='xMidYMid meet'
				className='pointer-events-none absolute inset-0 size-full'
				aria-hidden='true'
			>
				{layers.scene
					? overlay.scene.map((shape) => {
							const points = shape.points.map(([x, y]) => `${x * width},${y * height}`).join(' ')
							const color = SCENE_COLORS[shape.kind] ?? '#e5e7eb'
							return shape.points.length === 2 ? (
								<polyline key={shape.name} points={points} fill='none' stroke={color} strokeWidth={stroke * 1.5} />
							) : (
								<polygon
									key={shape.name}
									points={points}
									fill={color}
									fillOpacity={0.08}
									stroke={color}
									strokeOpacity={0.7}
									strokeWidth={stroke}
								/>
							)
						})
					: null}
				{layers.boxes && frame
					? frame.objects.map(([x1, y1, x2, y2, cls, id, labels], i) => {
							if (LIGHTS.has(cls) && labels.length === 0) return null
							if (layers.flaggedOnly && labels.length === 0) return null
							const flagged = labels.length > 0
							const color = flagged ? '#f43f5e' : cls === PERSON ? '#facc15' : '#38bdf8'
							const caption = flagged
								? labels.map((label) => classMeta[label as EventClass]?.title ?? label).join(', ')
								: `${DETECTOR_CLASSES[cls] ?? cls}${id >= 0 ? ` #${id}` : ''}`
							return (
								<g key={`${id}-${i}`}>
									<rect
										x={x1 * width}
										y={y1 * height}
										width={(x2 - x1) * width}
										height={(y2 - y1) * height}
										fill={flagged ? color : 'none'}
										fillOpacity={flagged ? 0.15 : 0}
										stroke={color}
										strokeWidth={flagged ? stroke * 1.6 : stroke}
									/>
									{flagged || (x2 - x1) * width > font * 3 ? (
										<text
											x={x1 * width}
											y={Math.max(font, y1 * height - stroke * 2)}
											fontSize={flagged ? font * 1.1 : font * 0.8}
											fontWeight={flagged ? 700 : 500}
											fill={color}
											stroke='#000'
											strokeWidth={font / 6}
											paintOrder='stroke'
										>
											{caption}
										</text>
									) : null}
								</g>
							)
						})
					: null}
			</svg>
			{riskNow !== null ? (
				<div
					className='pointer-events-none absolute top-3 right-3 rounded-md bg-black/70 px-2 py-1 font-mono text-xs text-white tabular-nums'
					title='Collision risk from pairwise time-to-collision (Part B)'
				>
					risk {riskNow.toFixed(2)}
				</div>
			) : null}
		</>
	)
}

export function OverlayControls({
	layers,
	onChange,
}: {
	layers: OverlayLayers
	onChange: (layers: OverlayLayers) => void
}) {
	const options: [keyof OverlayLayers, string][] = [
		['boxes', 'Boxes and tracks'],
		['flaggedOnly', 'Only flagged objects'],
		['scene', 'Scene geometry'],
	]
	return (
		<div className='flex flex-wrap items-center gap-x-4 gap-y-2 text-sm'>
			<label className='flex cursor-pointer items-center gap-2 font-medium'>
				<input
					type='checkbox'
					role='switch'
					className='size-4 accent-current'
					checked={layers.visible}
					onChange={(event) => onChange({ ...layers, visible: event.target.checked })}
				/>
				Show annotations
			</label>
			{options.map(([key, label]) => (
				<label
					key={key}
					className={cn('flex items-center gap-2', layers.visible ? 'cursor-pointer' : 'cursor-not-allowed opacity-50')}
				>
					<input
						type='checkbox'
						className='size-4 accent-current'
						checked={layers[key]}
						disabled={!layers.visible}
						onChange={(event) => onChange({ ...layers, [key]: event.target.checked })}
					/>
					{label}
				</label>
			))}
		</div>
	)
}
