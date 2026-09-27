'use client'

import { SignalStrip } from '@/components/charts/signal-strip'
import { TimeseriesChart, type Series } from '@/components/charts/timeseries-chart'
import { Pending } from '@/components/shared/pending'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { DETECTOR_CLASSES, type Sample } from '@/types/sample'

export const detectorSeries: Series[] = DETECTOR_CLASSES.map((key, index) => ({
	key,
	label: key.charAt(0).toUpperCase() + key.slice(1),
	color: `var(--series-${index + 1})`,
}))

function Figure({ src, alt, caption }: { src: string; alt: string; caption: string }) {
	return (
		<figure className='grid gap-2'>
			{/* eslint-disable-next-line @next/next/no-img-element -- static export, pre-sized JPEG */}
			<img src={src} alt={alt} loading='lazy' className='aspect-video w-full rounded-xl border object-cover' />
			<figcaption className='text-xs text-muted-foreground'>{caption}</figcaption>
		</figure>
	)
}

export function SampleEda({ sample }: { sample: Sample }) {
	const eda = sample.eda

	if (!eda) {
		return (
			<Pending title='EDA not generated yet'>
				Run <code className='font-mono'>web/scripts/build_sample_data.py</code> with the 1 fps frames for {sample.id}.
			</Pending>
		)
	}

	const brightness = eda.brightness.map(([t, value]) => ({ t, luma: value }))
	const motion = eda.motion.map(([t, value]) => ({ t, share: value * 100 }))

	return (
		<div className='grid gap-4 lg:grid-cols-2'>
			<Card className='lg:col-span-2'>
				<CardHeader>
					<CardTitle>Road users over time</CardTitle>
					<CardDescription>
						Detections per frame every {eda.step_sec} s ({eda.detector.weights}, {eda.detector.imgsz} px, confidence ≥{' '}
						{eda.detector.conf}). Counts are detections, not unique objects. Click a name to hide it.
					</CardDescription>
				</CardHeader>
				<CardContent>
					{eda.counts.length ? (
						<TimeseriesChart data={eda.counts} series={detectorSeries} height={240} valueFormat={(v) => v.toFixed(0)} />
					) : (
						<Pending title='Detector counts not generated' />
					)}
				</CardContent>
			</Card>

			{eda.signal.length ? (
				<Card className='lg:col-span-2'>
					<CardHeader>
						<CardTitle>Traffic-signal phases</CardTitle>
						<CardDescription>
							Seconds where the detector saw a red or green signal head. Several heads are visible at once, so both
							rows can be on together. This is what red_light and stop_line rules key on.
						</CardDescription>
					</CardHeader>
					<CardContent>
						<SignalStrip signal={eda.signal} duration={sample.video.duration_sec} step={eda.step_sec} />
					</CardContent>
				</Card>
			) : null}

			<Card>
				<CardHeader>
					<CardTitle>Lighting</CardTitle>
					<CardDescription>Mean frame brightness (luma, 0–255).</CardDescription>
				</CardHeader>
				<CardContent>
					<TimeseriesChart
						data={brightness}
						series={[{ key: 'luma', label: 'Brightness', color: 'var(--series-1)' }]}
						height={180}
						valueFormat={(v) => v.toFixed(0)}
					/>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Motion</CardTitle>
					<CardDescription>Share of the image that changed between frames (frame differencing, ~10 fps).</CardDescription>
				</CardHeader>
				<CardContent>
					{motion.length ? (
						<TimeseriesChart
							data={motion}
							series={[{ key: 'share', label: 'Moving area', color: 'var(--series-1)' }]}
							height={180}
							variant='area'
							valueFormat={(v) => `${v.toFixed(1)}%`}
						/>
					) : (
						<Pending title='Motion statistics not generated' />
					)}
				</CardContent>
			</Card>

			{eda.motion_heatmap_url ? (
				<Figure
					src={`/${eda.motion_heatmap_url}`}
					alt={`Motion heatmap for ${sample.id}`}
					caption='Motion heatmap: darker blue = pixels that change more often. Lanes, crossings and turning paths stand out.'
				/>
			) : null}
			{eda.flow_url ? (
				<Figure
					src={`/${eda.flow_url}`}
					alt={`Dominant motion direction for ${sample.id}`}
					caption='Dominant direction of motion (optical flow) per cell. This is how lane directions for wrong_way were checked.'
				/>
			) : null}
			{eda.positions_url ? (
				<Figure
					src={`/${eda.positions_url}`}
					alt={`Detection positions for ${sample.id}`}
					caption='Ground contact point of every detection: blue = pedestrians and cyclists, orange = motor vehicles.'
				/>
			) : null}
		</div>
	)
}
