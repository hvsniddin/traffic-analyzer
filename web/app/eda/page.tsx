import type { Metadata } from 'next'

import { EdaExplorer } from '@/components/samples/eda-explorer'
import { PageHeader, SectionTitle } from '@/components/shared/page-header'
import { summarize } from '@/lib/eda-summary'
import { formatSeconds } from '@/lib/format'
import { getSamples } from '@/lib/samples'

export const metadata: Metadata = { title: 'EDA' }

const dash = '—'
const fixed = (value: number | null, digits = 1) => (value === null ? dash : value.toFixed(digits))
const percent = (value: number | null) => (value === null ? dash : `${(value * 100).toFixed(0)}%`)

// Measured in docs/scene_alignment.md (median residual of stationary SIFT matches at 1280×720).
const alignment = [
	{ id: 'C3896', before: 0.29, after: 0.22, choice: 'Identity' },
	{ id: 'C3897', before: 0.21, after: 0.17, choice: 'Identity (reference clip)' },
	{ id: 'C3902', before: 38.35, after: 0.38, choice: 'Homography' },
	{ id: 'C3905', before: 15.04, after: 0.4, choice: 'Homography' },
]

export default async function EdaPage() {
	const samples = await getSamples()
	const summaries = samples.map(summarize)
	const totalSec = samples.reduce((sum, sample) => sum + sample.video.duration_sec, 0)
	const lumas = summaries.map((row) => row.meanLuma).filter((value): value is number => value !== null)
	const lumaRange = lumas.length > 1 ? [Math.min(...lumas), Math.max(...lumas)] : null

	return (
		<>
			<PageHeader kicker='Exploratory data analysis' title='What the four sample clips told us'>
				{samples.length} clips, {formatSeconds(totalSec)} of footage in total, all from one fixed 4K camera over a
				signalised junction with zebra crossings and a refuge island. Everything below is computed from the clips by{' '}
				<code className='font-mono text-sm'>web/scripts/build_sample_data.py</code>.
			</PageHeader>

			<section className='grid gap-2'>
				<SectionTitle title='Video properties'>
					Every clip is 4K H.264 at 140 Mbit/s and 29.97 fps. The task text says 25 fps, so frame timestamps must come
					from the file&apos;s real rate, never an assumed one. At 4K, decoding alone is the main runtime cost.
					{lumaRange ? (
						<>
							{' '}
							Lighting is not constant: mean brightness ranges from {lumaRange[0].toFixed(0)} to{' '}
							{lumaRange[1].toFixed(0)} across clips, so detection thresholds need checking on the darkest clip,
							not only the brightest.
						</>
					) : null}
				</SectionTitle>
				<div className='overflow-x-auto rounded-xl border'>
					<table className='w-full text-sm'>
						<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
							<tr>
								<th className='px-3 py-2 font-medium'>Clip</th>
								<th className='px-3 py-2 font-medium'>Resolution</th>
								<th className='px-3 py-2 text-right font-medium'>fps</th>
								<th className='px-3 py-2 text-right font-medium'>Length</th>
								<th className='px-3 py-2 text-right font-medium'>Frames</th>
								<th className='px-3 py-2 text-right font-medium'>Size</th>
								<th className='px-3 py-2 text-right font-medium'>Brightness</th>
							</tr>
						</thead>
						<tbody className='font-mono tabular-nums'>
							{samples.map((sample, index) => (
								<tr key={sample.id} className='border-t'>
									<td className='px-3 py-2 font-semibold'>{sample.id}</td>
									<td className='px-3 py-2'>
										{sample.video.width}×{sample.video.height}
									</td>
									<td className='px-3 py-2 text-right'>{sample.video.fps}</td>
									<td className='px-3 py-2 text-right'>{sample.video.duration_sec.toFixed(1)} s</td>
									<td className='px-3 py-2 text-right'>{sample.video.n_frames.toLocaleString('en-US')}</td>
									<td className='px-3 py-2 text-right'>{sample.video.size_gb} GB</td>
									<td className='px-3 py-2 text-right'>{fixed(summaries[index].meanLuma, 0)}</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Traffic and pedestrians per clip'>
					Detections per sampled frame from our fine-tuned YOLO. &ldquo;Red seen&rdquo; is the share of sampled seconds
					with at least one red signal head detected.
				</SectionTitle>
				<div className='overflow-x-auto rounded-xl border'>
					<table className='w-full text-sm'>
						<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
							<tr>
								<th className='px-3 py-2 font-medium'>Clip</th>
								<th className='px-3 py-2 text-right font-medium'>Vehicles, mean</th>
								<th className='px-3 py-2 text-right font-medium'>Vehicles, peak</th>
								<th className='px-3 py-2 text-right font-medium'>People, mean</th>
								<th className='px-3 py-2 text-right font-medium'>People, peak</th>
								<th className='px-3 py-2 text-right font-medium'>Red seen</th>
								<th className='px-3 py-2 text-right font-medium'>Moving area</th>
							</tr>
						</thead>
						<tbody className='font-mono tabular-nums'>
							{summaries.map((row) => (
								<tr key={row.id} className='border-t'>
									<td className='px-3 py-2 font-semibold'>{row.id}</td>
									<td className='px-3 py-2 text-right'>{fixed(row.meanVehicles)}</td>
									<td className='px-3 py-2 text-right'>{fixed(row.peakVehicles, 0)}</td>
									<td className='px-3 py-2 text-right'>{fixed(row.meanPeople)}</td>
									<td className='px-3 py-2 text-right'>{fixed(row.peakPeople, 0)}</td>
									<td className='px-3 py-2 text-right'>{percent(row.redShare)}</td>
									<td className='px-3 py-2 text-right'>
										{row.movingShare === null ? dash : `${(row.movingShare * 100).toFixed(1)}%`}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Finding: the “fixed” camera is not pixel-identical across clips'>
					Our scene polygons (lanes, stop lines, crossings, island) were drawn on C3897. On C3902 the same features sit
					38 px away at 1280×720, about 115 px in the original 4K. Unaligned crossing polygons overlap the aligned ones by
					only 27–47% on C3902, and in a spot check alignment changed the crossing membership of 111 of 361 person
					detections, so point-in-zone rules would have fired in the wrong place. We now estimate one
					homography per video from stationary SIFT features before any rule runs.
				</SectionTitle>
				<div className='overflow-x-auto rounded-xl border'>
					<table className='w-full text-sm'>
						<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
							<tr>
								<th className='px-3 py-2 font-medium'>Clip</th>
								<th className='px-3 py-2 text-right font-medium'>Residual before</th>
								<th className='px-3 py-2 text-right font-medium'>Residual after</th>
								<th className='px-3 py-2 font-medium'>Transform used</th>
							</tr>
						</thead>
						<tbody>
							{alignment.map((row) => (
								<tr key={row.id} className='border-t'>
									<td className='px-3 py-2 font-mono font-semibold'>{row.id}</td>
									<td className='px-3 py-2 text-right font-mono tabular-nums'>{row.before.toFixed(2)} px</td>
									<td className='px-3 py-2 text-right font-mono tabular-nums'>{row.after.toFixed(2)} px</td>
									<td className='px-3 py-2'>{row.choice}</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
				<div className='mt-2 grid gap-4 md:grid-cols-2'>
					{['C3902', 'C3905'].map((id) => (
						<figure key={id} className='grid gap-2'>
							{/* eslint-disable-next-line @next/next/no-img-element -- static export */}
							<img
								src={`/eda/alignment_${id}.jpg`}
								alt={`Scene polygons on ${id} before and after alignment`}
								loading='lazy'
								className='w-full rounded-xl border'
							/>
							<figcaption className='text-xs text-muted-foreground'>
								{id}: scene polygons before alignment (red) and after (green).
							</figcaption>
						</figure>
					))}
				</div>
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Per-clip time series and heatmaps'>
					Counts by class over time, signal phases, lighting, motion, lane directions and where road users actually walk
					and drive.
				</SectionTitle>
				<EdaExplorer samples={samples} />
			</section>
		</>
	)
}
