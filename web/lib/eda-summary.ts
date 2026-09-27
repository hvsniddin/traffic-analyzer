import type { Sample } from '@/types/sample'

export type EdaSummary = {
	id: string
	meanVehicles: number | null
	peakVehicles: number | null
	meanPeople: number | null
	peakPeople: number | null
	meanLuma: number | null
	redShare: number | null
	movingShare: number | null
}

const mean = (values: number[]) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null)

/** Per-clip figures derived from the exported EDA series; null when not generated. */
export function summarize(sample: Sample): EdaSummary {
	const eda = sample.eda
	const counts = eda?.counts ?? []
	const vehicles = counts.map((row) => row.car + row.bus + row.truck + row.motorcycle)
	const people = counts.map((row) => row.person + row.bicycle)
	const signal = eda?.signal ?? []

	return {
		id: sample.id,
		meanVehicles: mean(vehicles),
		peakVehicles: vehicles.length ? Math.max(...vehicles) : null,
		meanPeople: mean(people),
		peakPeople: people.length ? Math.max(...people) : null,
		meanLuma: mean(eda?.brightness.map(([, luma]) => luma) ?? []),
		redShare: signal.length ? signal.filter((row) => row.red > 0).length / signal.length : null,
		movingShare: mean(eda?.motion.map(([, share]) => share) ?? []),
	}
}
