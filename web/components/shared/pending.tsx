import { HourglassIcon } from 'lucide-react'
import * as React from 'react'

import { cn } from '@/lib/utils'

/** Placeholder for a result the pipeline has not produced yet. */
export function Pending({ title, children, className }: { title: string; children?: React.ReactNode; className?: string }) {
	return (
		<div className={cn('flex items-start gap-3 rounded-xl border border-dashed p-4 text-sm', className)}>
			<HourglassIcon className='mt-0.5 size-4 shrink-0 text-muted-foreground' />
			<div className='grid gap-1'>
				<p className='font-medium'>{title}</p>
				{children ? <div className='text-muted-foreground'>{children}</div> : null}
			</div>
		</div>
	)
}
