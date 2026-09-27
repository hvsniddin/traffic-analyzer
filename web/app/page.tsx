import { ArrowRightIcon } from 'lucide-react'
import Link from 'next/link'

import { SampleCard } from '@/components/samples/sample-card'
import { SectionTitle } from '@/components/shared/page-header'
import { PipelineDiagram } from '@/components/shared/pipeline-diagram'
import { buttonClass } from '@/components/ui/button'
import { classApproach, pipeline } from '@/content/approach'
import { getSamples } from '@/lib/samples'
import { EVENT_CLASSES } from '@/types/events'

export default async function HomePage() {
	const samples = await getSamples()
	const implemented = EVENT_CLASSES.filter((label) => classApproach[label].status === 'rule').length

	const stats = [
		{ value: String(samples.length), label: 'sample clips analysed' },
		{ value: `${implemented} / 14`, label: 'event classes detected' },
		{ value: '4K · 29.97', label: 'camera resolution and fps' },
		{ value: '< 0.5 px', label: 'scene-map alignment residual' },
	]

	return (
		<>
			<section className='grid gap-6 pt-14 pb-12 sm:pt-20'>
				<p className='font-mono text-xs tracking-[0.18em] text-muted-foreground uppercase'>
					WIUT Hackathon 2026 · Computer Vision track
				</p>
				<h1 className='max-w-4xl text-4xl font-semibold tracking-tight text-balance sm:text-5xl'>
					Traffic events from a fixed road camera, as time segments you can click.
				</h1>
				<p className='max-w-2xl text-lg text-muted-foreground'>
					We detect and track every road user, align a hand-drawn map of the junction to each clip, and turn trajectories
					into events such as jaywalking, red-light running and near misses.
				</p>
				<div className='flex flex-wrap gap-3'>
					<Link href='/demo/' className={buttonClass()}>
						Try the live demo <ArrowRightIcon />
					</Link>
					<Link href='/samples/' className={buttonClass('outline')}>
						See sample results
					</Link>
				</div>
			</section>

			<dl className='grid grid-cols-2 gap-px overflow-hidden rounded-2xl border bg-border lg:grid-cols-4'>
				{stats.map((stat) => (
					<div key={stat.label} className='grid gap-1 bg-card p-5'>
						<dt className='order-2 text-sm text-muted-foreground'>{stat.label}</dt>
						<dd className='order-1 font-mono text-2xl font-semibold tabular-nums'>{stat.value}</dd>
					</div>
				))}
			</dl>

			<section className='mt-16 grid gap-2'>
				<SectionTitle title='How it works'>
					One learned component, everything after it deterministic.{' '}
					<Link href='/approach/' className='text-primary hover:underline'>
						Read the full approach
					</Link>
					.
				</SectionTitle>
				<PipelineDiagram stages={pipeline} />
			</section>

			<section className='mt-16 grid gap-2'>
				<SectionTitle title='Sample videos'>Playback, event timeline and EDA for every clip the organizers gave us.</SectionTitle>
				<div className='grid gap-4 sm:grid-cols-2 lg:grid-cols-4'>
					{samples.map((sample) => (
						<SampleCard key={sample.id} sample={sample} />
					))}
				</div>
			</section>
		</>
	)
}
