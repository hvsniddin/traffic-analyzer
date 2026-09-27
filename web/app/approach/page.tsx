import type { Metadata } from 'next'

import { PageHeader, SectionTitle } from '@/components/shared/page-header'
import { PipelineDiagram } from '@/components/shared/pipeline-diagram'
import { Badge } from '@/components/ui/badge'
import { classApproach, pipeline } from '@/content/approach'
import { classMeta, classStyle } from '@/lib/events'
import { cn } from '@/lib/utils'
import { EVENT_CLASSES } from '@/types/events'

export const metadata: Metadata = { title: 'Problem and approach' }

export default function ApproachPage() {
	const implemented = EVENT_CLASSES.filter((label) => classApproach[label].status === 'rule')

	return (
		<>
			<PageHeader kicker='Problem and approach' title='Detector, tracker, aligned scene map, rules'>
				The task: given a clip from one fixed junction camera, return every traffic event as{' '}
				<code className='font-mono text-sm'>[start_sec, end_sec, label]</code> over 14 classes, scored by temporal-IoU
				F1 at 0.3, 0.5 and 0.7. We manually labelled event intervals in all four sample clips and fine-tuned the
				object detector on annotated frames. Traffic rules use tracked objects and a scene map.
			</PageHeader>

			<section className='grid gap-2'>
				<SectionTitle title='Pipeline' />
				<PipelineDiagram stages={pipeline} />
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='What is learned and what is rule-based'>
					Learned: the object detector only. Rule-based: tracking association, scene alignment, every event decision
					and segment post-processing. Rules are inspectable and need no event labels, and the camera angle is fixed, so
					one scene map, re-aligned per clip, covers the hidden test set.
				</SectionTitle>
				<div className='grid gap-4 text-sm md:grid-cols-2'>
					<div className='grid gap-2 rounded-2xl border p-5'>
						<p className='font-semibold'>Models and data</p>
						<ul className='grid list-disc gap-1.5 ps-5 text-muted-foreground'>
							<li>
								<span className='text-foreground'>YOLO (Ultralytics), fine-tuned</span>: 8 classes: car, bus, truck,
								motorcycle, bicycle, person, red light, green light. Weights ship in{' '}
								<code className='font-mono'>weights/</code>. Signal heads as detector classes let rules know the phase
								without a separate classifier.
							</li>
							<li>
								<span className='text-foreground'>ByteTrack</span> (via Ultralytics) links detections into tracks,
								which every rule needs for direction, speed and dwell time.
							</li>
							<li>
								<span className='text-foreground'>OpenCV SIFT + RANSAC</span> estimates one homography per clip from
								stationary features against a bundled reference frame.
							</li>
							<li>Fine-tuning used annotated frames from the four organizer-provided videos only. No public dataset was added.</li>
						</ul>
					</div>
					<div className='grid gap-2 rounded-2xl border p-5'>
						<p className='font-semibold'>Why this design</p>
						<ul className='grid list-disc gap-1.5 ps-5 text-muted-foreground'>
							<li>Four manually labelled clips give us a small dev set for tuning, while explicit rules use the scene layout directly.</li>
							<li>
								The hidden set uses the same camera and angle, so a hand-drawn map of lanes, stop lines, crossings and
								the island transfers, once alignment absorbs the small framing shifts we measured.
							</li>
							<li>
								Boundaries decide the score at IoU 0.7. Rules give exact start and end frames tied to the task&apos;s
								own conventions (for example red_light starts when the front crosses the stop line).
							</li>
							<li>One detector pass shares tracks across all implemented rules, reducing repeated inference.</li>
						</ul>
					</div>
				</div>
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Per-class logic'>
					{implemented.length} of 14 classes have a detector today. The others are listed with the approach we would try
					next. The harness only scores classes we predict or that occur, so we do not emit guesses.
				</SectionTitle>
				<div className='overflow-x-auto rounded-xl border'>
					<table className='w-full text-sm'>
						<thead className='bg-muted/60 text-left text-xs text-muted-foreground'>
							<tr>
								<th className='px-3 py-2 font-medium'>Class</th>
								<th className='px-3 py-2 font-medium'>Status</th>
								<th className='px-3 py-2 font-medium'>How it is decided</th>
							</tr>
						</thead>
						<tbody>
							{EVENT_CLASSES.map((label) => {
								const approach = classApproach[label]
								return (
									<tr key={label} className={cn('border-t align-top', approach.status === 'planned' && 'text-muted-foreground')}>
										<td className='px-3 py-2.5'>
											<span className='flex items-center gap-2 font-medium text-foreground'>
												<span className={cn('size-2 shrink-0 rounded-full', classStyle(label).dot)} />
												{classMeta[label].title}
											</span>
											<span className='ps-4 font-mono text-xs text-muted-foreground'>{label}</span>
										</td>
										<td className='px-3 py-2.5'>
											{approach.status === 'rule' ? (
												<Badge variant='success'>Rule</Badge>
											) : (
												<Badge variant='outline'>Not yet</Badge>
											)}
										</td>
										<td className='px-3 py-2.5'>{approach.how}</td>
									</tr>
								)
							})}
						</tbody>
					</table>
				</div>
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Scene map'>
					Polygons and lines in normalised coordinates, drawn once on C3897 at 287 s and stored in{' '}
					<code className='font-mono'>scene.json</code>: carriageway, lanes with travel direction, crossings, refuge
					island, kerb waiting zones, stop lines and solid markings.
				</SectionTitle>
				<figure className='grid gap-2'>
					{/* eslint-disable-next-line @next/next/no-img-element -- static export */}
					<img src='/eda/scene_layout.jpg' alt='Scene map overlaid on the camera view' className='w-full rounded-xl border' />
					<figcaption className='text-xs text-muted-foreground'>Scene map drawn over the reference frame.</figcaption>
				</figure>
			</section>
		</>
	)
}
