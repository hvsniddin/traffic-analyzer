import * as React from 'react'

import { cn } from '@/lib/utils'

const variants = {
	default: 'bg-muted text-foreground',
	outline: 'border text-muted-foreground',
	primary: 'bg-primary/10 text-primary',
	success: 'bg-success/10 text-success',
	destructive: 'bg-destructive/10 text-destructive',
} as const

type BadgeProps = React.ComponentProps<'span'> & { variant?: keyof typeof variants }

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
	return (
		<span
			className={cn(
				'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium whitespace-nowrap [&_svg]:size-3',
				variants[variant],
				className,
			)}
			{...props}
		/>
	)
}
