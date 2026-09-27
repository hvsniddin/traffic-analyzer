import * as React from 'react'

export function PageHeader({
	kicker,
	title,
	children,
}: {
	kicker: string
	title: string
	children?: React.ReactNode
}) {
	return (
		<div className='grid max-w-3xl gap-3 pt-10 pb-8 sm:pt-14'>
			<p className='font-mono text-xs tracking-[0.18em] text-muted-foreground uppercase'>{kicker}</p>
			<h1 className='text-3xl font-semibold tracking-tight text-balance sm:text-4xl'>{title}</h1>
			{children ? <div className='text-base leading-relaxed text-muted-foreground'>{children}</div> : null}
		</div>
	)
}

export function SectionTitle({ title, children }: { title: string; children?: React.ReactNode }) {
	return (
		<div className='grid gap-1 pb-4'>
			<h2 className='text-xl font-semibold tracking-tight'>{title}</h2>
			{children ? <p className='max-w-3xl text-sm text-muted-foreground'>{children}</p> : null}
		</div>
	)
}
