import * as React from 'react'

import { cn } from '@/lib/utils'

const variants = {
	default: 'bg-primary text-primary-foreground hover:bg-primary/90',
	outline: 'border bg-card hover:bg-muted',
	ghost: 'hover:bg-muted',
} as const

const sizes = {
	default: 'h-9 px-4 text-sm',
	sm: 'h-8 px-3 text-xs',
	icon: 'size-9',
} as const

type ButtonProps = React.ComponentProps<'button'> & {
	variant?: keyof typeof variants
	size?: keyof typeof sizes
}

export function buttonClass(variant: keyof typeof variants = 'default', size: keyof typeof sizes = 'default') {
	return cn(
		'inline-flex shrink-0 items-center justify-center gap-2 rounded-lg font-medium whitespace-nowrap transition-colors outline-none focus-visible:ring-2 focus-visible:ring-primary/50 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0',
		variants[variant],
		sizes[size],
	)
}

export function Button({ className, variant, size, type = 'button', ...props }: ButtonProps) {
	return <button type={type} className={cn(buttonClass(variant, size), className)} {...props} />
}
