import { TriangleAlertIcon } from 'lucide-react'
import * as React from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

type ErrorStateProps = Omit<React.ComponentProps<'div'>, 'title'> & {
	title?: string
	description?: string
	retryLabel?: string
	onRetry?: () => void
}

export function ErrorState({
	title = 'Something went wrong',
	description,
	retryLabel = 'Try again',
	onRetry,
	className,
	children,
	...props
}: ErrorStateProps) {
	return (
		<div
			role='alert'
			className={cn('flex flex-col items-center justify-center gap-4 p-8 text-center', className)}
			{...props}
		>
			<span className='flex size-11 items-center justify-center rounded-full bg-destructive/10 text-destructive'>
				<TriangleAlertIcon className='size-5' />
			</span>
			<div className='grid gap-1.5'>
				<p className='font-medium'>{title}</p>
				{description ? <p className='max-w-md text-sm text-muted-foreground'>{description}</p> : null}
			</div>
			{children}
			{onRetry ? (
				<Button variant='outline' onClick={onRetry}>
					{retryLabel}
				</Button>
			) : null}
		</div>
	)
}

export function InlineError({ className, children, ...props }: React.ComponentProps<'p'>) {
	return (
		<p role='alert' className={cn('flex items-start gap-2 text-sm text-destructive', className)} {...props}>
			<TriangleAlertIcon className='mt-0.5 size-4 shrink-0' />
			<span>{children}</span>
		</p>
	)
}
