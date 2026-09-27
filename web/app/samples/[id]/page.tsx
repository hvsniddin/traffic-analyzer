import { ArrowLeftIcon } from 'lucide-react'
import type { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'

import { SampleEda } from '@/components/samples/sample-eda'
import { SampleResults } from '@/components/samples/sample-results'
import { PageHeader, SectionTitle } from '@/components/shared/page-header'
import { Pending } from '@/components/shared/pending'
import { formatSeconds, formatTimestamp } from '@/lib/format'
import { getSample, getSampleIds } from '@/lib/samples'

type Params = { params: Promise<{ id: string }> }

export const dynamicParams = false

export async function generateStaticParams() {
	return (await getSampleIds()).map((id) => ({ id }))
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
	const { id } = await params
	return { title: `Sample ${id}` }
}

export default async function SamplePage({ params }: Params) {
	const { id } = await params
	const sample = await getSample(id)
	if (!sample) notFound()

	const { video } = sample
	const facts = [
		['Resolution', `${video.width} × ${video.height}`],
		['Frame rate', `${video.fps} fps`],
		['Length', formatSeconds(video.duration_sec)],
		['Frames', video.n_frames.toLocaleString('en-US')],
		['Codec', `${video.codec.toUpperCase()}, ${video.bitrate_mbps} Mbit/s`],
		['File', `${video.file}, ${video.size_gb} GB`],
	]

	return (
		<>
			<Link
				href='/samples/'
				className='mt-8 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground'
			>
				<ArrowLeftIcon className='size-4' /> All samples
			</Link>
			<PageHeader kicker='Sample result' title={sample.id} />

			<dl className='mb-10 grid grid-cols-2 gap-x-6 gap-y-3 rounded-2xl border bg-card p-5 text-sm sm:grid-cols-3 lg:grid-cols-6'>
				{facts.map(([label, value]) => (
					<div key={label} className='grid gap-0.5'>
						<dt className='text-xs text-muted-foreground'>{label}</dt>
						<dd className='font-medium'>{value}</dd>
					</div>
				))}
			</dl>

			<section className='grid gap-2'>
				<SectionTitle title='Playback and event timeline'>
					Click an event to seek. Where the model and our labels disagree, compare the two lanes directly.
				</SectionTitle>
				<SampleResults sample={sample} />
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='Failure cases'>
					Where the output is wrong on this clip and why. Stated plainly so the gaps are visible.
				</SectionTitle>
				{sample.failures.length ? (
					<ul className='grid gap-3'>
						{sample.failures.map((failure) => (
							<li key={`${failure.start_sec}-${failure.end_sec}`} className='rounded-xl border p-4 text-sm'>
								<span className='font-mono text-xs text-muted-foreground'>
									{formatTimestamp(failure.start_sec)}–{formatTimestamp(failure.end_sec)}
								</span>
								<p>{failure.note}</p>
							</li>
						))}
					</ul>
				) : (
					<Pending title='Not reviewed yet'>
						Failure cases are written after comparing predictions against the dev labels for this clip.
					</Pending>
				)}
			</section>

			<section className='mt-14 grid gap-2'>
				<SectionTitle title='EDA for this clip' />
				<SampleEda sample={sample} />
			</section>
		</>
	)
}
