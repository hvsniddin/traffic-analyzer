'use client'

import * as React from 'react'
import {
	Area,
	AreaChart,
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from 'recharts'

import { formatTimestamp } from '@/lib/format'
import { cn } from '@/lib/utils'

export type Series = { key: string; label: string; color: string }

type TimeseriesChartProps = {
	data: Record<string, number>[]
	series: Series[]
	height?: number
	yLabel?: string
	yDomain?: [number, number]
	valueFormat?: (value: number) => string
	variant?: 'line' | 'area'
	onPointClick?: (t: number) => void
}

const axisProps = {
	stroke: 'var(--muted-foreground)',
	fontSize: 11,
	tickLine: false,
	axisLine: false,
} as const

/** Time on x (key "t", seconds), one shared y axis. Legend toggles series. */
export function TimeseriesChart({
	data,
	series,
	height = 220,
	yLabel,
	yDomain,
	valueFormat = (value) => value.toLocaleString('en-US', { maximumFractionDigits: 2 }),
	variant = 'line',
	onPointClick,
}: TimeseriesChartProps) {
	const [hidden, setHidden] = React.useState<Set<string>>(new Set())
	const visible = series.filter((item) => !hidden.has(item.key))

	function toggle(key: string) {
		setHidden((current) => {
			const next = new Set(current)
			if (next.has(key)) next.delete(key)
			else if (next.size < series.length - 1) next.add(key)
			return next
		})
	}

	const tooltip = (
		<Tooltip
			cursor={{ stroke: 'var(--muted-foreground)', strokeWidth: 1 }}
			content={({ active, payload, label }) =>
				active && payload?.length ? (
					<div className='rounded-lg border bg-card px-3 py-2 text-xs shadow-lg'>
						<p className='mb-1 font-mono text-muted-foreground'>{formatTimestamp(Number(label), 0)}</p>
						{payload.map((entry) => (
							<p key={String(entry.dataKey)} className='flex items-center gap-2'>
								<span className='size-2 rounded-full' style={{ background: entry.color }} />
								<span>{series.find((item) => item.key === entry.dataKey)?.label}</span>
								<span className='ms-auto ps-3 font-mono tabular-nums'>{valueFormat(Number(entry.value))}</span>
							</p>
						))}
					</div>
				) : null
			}
		/>
	)

	const common = {
		data,
		margin: { top: 8, right: 8, bottom: 0, left: 0 },
		onClick: onPointClick
			? (state: { activeLabel?: string | number }) => {
					if (state?.activeLabel !== undefined) onPointClick(Number(state.activeLabel))
				}
			: undefined,
	}

	const axes = (
		<>
			<CartesianGrid vertical={false} stroke='var(--grid)' />
			<XAxis
				dataKey='t'
				type='number'
				domain={['dataMin', 'dataMax']}
				tickFormatter={(value) => formatTimestamp(value, 0)}
				{...axisProps}
			/>
			<YAxis
				width={40}
				domain={yDomain ?? [0, 'auto']}
				tickFormatter={(value) => valueFormat(value)}
				label={
					yLabel
						? { value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--muted-foreground)' }
						: undefined
				}
				{...axisProps}
			/>
			{tooltip}
		</>
	)

	return (
		<div className='grid gap-3'>
			{series.length > 1 ? (
				<div className='flex flex-wrap gap-x-4 gap-y-1'>
					{series.map((item) => (
						<button
							key={item.key}
							type='button'
							onClick={() => toggle(item.key)}
							aria-pressed={!hidden.has(item.key)}
							className={cn(
								'flex items-center gap-1.5 text-xs transition-opacity',
								hidden.has(item.key) && 'opacity-40',
							)}
						>
							<span className='h-0.5 w-3 rounded-full' style={{ background: item.color }} />
							{item.label}
						</button>
					))}
				</div>
			) : null}
			<div style={{ height }}>
				<ResponsiveContainer width='100%' height='100%'>
					{variant === 'area' ? (
						<AreaChart {...common}>
							{axes}
							{visible.map((item) => (
								<Area
									key={item.key}
									dataKey={item.key}
									type='monotone'
									stroke={item.color}
									strokeWidth={2}
									fill={item.color}
									fillOpacity={0.12}
									dot={false}
									isAnimationActive={false}
								/>
							))}
						</AreaChart>
					) : (
						<LineChart {...common}>
							{axes}
							{visible.map((item) => (
								<Line
									key={item.key}
									dataKey={item.key}
									type='monotone'
									stroke={item.color}
									strokeWidth={2}
									dot={false}
									activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--card)' }}
									isAnimationActive={false}
								/>
							))}
						</LineChart>
					)}
				</ResponsiveContainer>
			</div>
		</div>
	)
}
