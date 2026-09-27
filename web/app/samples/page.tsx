import type { Metadata } from 'next'

import { SampleCard } from '@/components/samples/sample-card'
import { PageHeader } from '@/components/shared/page-header'
import { getSamples } from '@/lib/samples'

export const metadata: Metadata = { title: 'Results on the samples' }

export default async function SamplesPage() {
	const samples = await getSamples()

	return (
		<>
			<PageHeader kicker='Results' title='Every sample video, with its event timeline'>
				The organizers supplied {samples.length} unlabelled clips from the camera. Each page shows the playback, the
				timeline of predicted events next to our own dev labels, per-video EDA, and the failure cases we found.
			</PageHeader>
			<div className='grid gap-4 sm:grid-cols-2 lg:grid-cols-4'>
				{samples.map((sample) => (
					<SampleCard key={sample.id} sample={sample} />
				))}
			</div>
		</>
	)
}
