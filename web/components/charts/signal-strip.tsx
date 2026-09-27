import { formatTimestamp } from '@/lib/format'
import type { SignalRow } from '@/types/sample'

// Rows (not colour) carry identity, so red/green stays readable under CVD.
const rows = [
	{ key: 'red' as const, label: 'Red light seen', color: '#e34948' },
	{ key: 'green' as const, label: 'Green light seen', color: '#008300' },
]

export function SignalStrip({ signal, duration, step }: { signal: SignalRow[]; duration: number; step: number }) {
	if (signal.length === 0) return null

	return (
		<div className='grid gap-1.5'>
			{rows.map((row) => (
				<div key={row.key} className='grid grid-cols-[7.5rem_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[9rem_minmax(0,1fr)]'>
					<span className='text-xs'>{row.label}</span>
					<div className='relative h-5 overflow-hidden rounded-md bg-muted'>
						{signal
							.filter((sample) => sample[row.key] > 0)
							.map((sample) => (
								<span
									key={sample.t}
									title={`${formatTimestamp(sample.t, 0)}: ${sample[row.key]} detected`}
									className='absolute inset-y-0.5'
									style={{
										left: `${(sample.t / duration) * 100}%`,
										width: `calc(${(step / duration) * 100}% - 1px)`,
										background: row.color,
									}}
								/>
							))}
					</div>
				</div>
			))}
		</div>
	)
}
