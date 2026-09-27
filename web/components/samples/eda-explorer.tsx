'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'
import type { Sample } from '@/types/sample'

import { SampleEda } from './sample-eda'

export function EdaExplorer({ samples }: { samples: Sample[] }) {
	const [selected, setSelected] = React.useState(samples[0]?.id)
	const sample = samples.find((item) => item.id === selected)

	return (
		<div className='grid gap-4'>
			<div role='tablist' aria-label='Sample video' className='flex flex-wrap gap-1'>
				{samples.map((item) => (
					<button
						key={item.id}
						role='tab'
						aria-selected={item.id === selected}
						onClick={() => setSelected(item.id)}
						className={cn(
							'rounded-md px-3 py-1.5 font-mono text-sm',
							item.id === selected ? 'bg-muted font-semibold' : 'text-muted-foreground hover:text-foreground',
						)}
					>
						{item.id}
					</button>
				))}
			</div>
			{sample ? <SampleEda key={sample.id} sample={sample} /> : null}
		</div>
	)
}
