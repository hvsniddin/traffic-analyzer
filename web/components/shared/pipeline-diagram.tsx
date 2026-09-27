import { ArrowDownIcon, ArrowRightIcon } from 'lucide-react'
import { Fragment } from 'react'

import type { Stage } from '@/content/approach'
import { cn } from '@/lib/utils'

const kindStyle = {
	learned: 'border-primary/50 bg-primary/5',
	rule: 'bg-card',
	io: 'border-dashed bg-muted/40',
} as const

const kindLabel = { learned: 'Learned', rule: 'Rule-based', io: '' } as const

/** Wraps horizontally on wide screens, stacks on phones. */
export function PipelineDiagram({ stages }: { stages: Stage[] }) {
	return (
		<div className='grid gap-4'>
			<ol className='flex flex-col items-stretch gap-2 lg:flex-row lg:flex-wrap lg:items-center'>
				{stages.map((stage, index) => (
					<Fragment key={stage.title}>
						<li className={cn('grid gap-1 rounded-xl border p-3 lg:w-[calc(25%-2.25rem)]', kindStyle[stage.kind])}>
							<div className='flex items-center justify-between gap-2'>
								<span className='text-sm font-semibold'>{stage.title}</span>
								{kindLabel[stage.kind] ? (
									<span
										className={cn(
											'text-[10px] font-medium tracking-wide uppercase',
											stage.kind === 'learned' ? 'text-primary' : 'text-muted-foreground',
										)}
									>
										{kindLabel[stage.kind]}
									</span>
								) : null}
							</div>
							<span className='text-xs text-muted-foreground'>{stage.detail}</span>
						</li>
						{index < stages.length - 1 ? (
							<li aria-hidden className='flex justify-center text-muted-foreground'>
								<ArrowDownIcon className='size-4 lg:hidden' />
								<ArrowRightIcon className='hidden size-4 lg:block' />
							</li>
						) : null}
					</Fragment>
				))}
			</ol>
			<p className='text-xs text-muted-foreground'>
				Blue border = learned component. Everything else is deterministic code whose thresholds are listed below.
			</p>
		</div>
	)
}
