import { EVENT_CLASSES, type EventClass, type TrafficEvent } from '@/types/events'

// Classes share a colour by family; the lane label carries identity, so colour
// is never the only cue. Palette validated for CVD in light and dark modes.
export type EventFamily = 'pedestrian' | 'collision' | 'violation' | 'flow'

export const familyLabels: Record<EventFamily, string> = {
	pedestrian: 'Pedestrian safety',
	collision: 'Collision risk',
	violation: 'Traffic-rule violation',
	flow: 'Flow and road state',
}

type ClassMeta = {
	title: string
	family: EventFamily
	definition: string
	start: string
	end: string
}

// Definitions and boundary conventions from the organizers' task page.
export const classMeta: Record<EventClass, ClassMeta> = {
	accident: {
		title: 'Collision',
		family: 'collision',
		definition: 'Contact between two or more road users, or a road user and a fixed object.',
		start: 'First frame where contact is visible',
		end: 'All involved objects stop moving or leave the frame',
	},
	near_miss: {
		title: 'Near miss',
		family: 'collision',
		definition: 'Sharp braking or swerving to avoid a collision; no contact.',
		start: 'Onset of the evasive action',
		end: 'Road users are clear of each other',
	},
	red_light: {
		title: 'Red-light running',
		family: 'violation',
		definition: 'A vehicle crosses the stop line while its signal is red.',
		start: 'Front of the vehicle crosses the stop line',
		end: 'Vehicle leaves the intersection or the frame',
	},
	wrong_way: {
		title: 'Wrong-way driving',
		family: 'violation',
		definition: 'A vehicle moves against the traffic direction of its lane.',
		start: 'Vehicle enters the opposing lane',
		end: 'Vehicle returns to a correct lane or leaves the frame',
	},
	illegal_u_turn: {
		title: 'Illegal U-turn',
		family: 'violation',
		definition: 'A U-turn where the road markings or signs prohibit it.',
		start: 'Vehicle starts turning',
		end: 'Vehicle completes the turn',
	},
	stopped_vehicle: {
		title: 'Stopped vehicle',
		family: 'flow',
		definition: 'Stationary on the carriageway for 10 s or more, not queued at a signal.',
		start: 'Vehicle stops',
		end: 'Vehicle moves again or is removed',
	},
	jaywalking: {
		title: 'Pedestrian on roadway',
		family: 'pedestrian',
		definition: 'A pedestrian on the carriageway outside a crossing.',
		start: 'Pedestrian steps onto the road',
		end: 'Pedestrian leaves the road',
	},
	failure_to_yield: {
		title: 'Not yielding to a pedestrian',
		family: 'pedestrian',
		definition: 'A vehicle drives through a crossing while a pedestrian is on it or stepping onto it.',
		start: 'Vehicle enters the crossing',
		end: 'Vehicle leaves the crossing',
	},
	illegal_turn: {
		title: 'Illegal turn',
		family: 'violation',
		definition: 'A turn from the wrong lane or in a prohibited direction.',
		start: 'Vehicle starts turning',
		end: 'Vehicle completes the turn',
	},
	solid_line_crossing: {
		title: 'Solid line crossing',
		family: 'violation',
		definition: 'A lane change or manoeuvre across a solid marking.',
		start: 'Wheel crosses the line',
		end: 'Vehicle is fully in the new lane',
	},
	stop_line: {
		title: 'Stop-line violation',
		family: 'violation',
		definition: 'A vehicle stops past the stop line on red without entering the intersection.',
		start: 'Vehicle stops',
		end: 'Signal turns green',
	},
	congestion: {
		title: 'Congestion',
		family: 'flow',
		definition: 'Traffic at a standstill or crawling across all lanes of a direction.',
		start: 'Queue stops moving',
		end: 'Queue clears',
	},
	road_obstacle: {
		title: 'Obstacle on road',
		family: 'flow',
		definition: 'Debris, animal, or fallen object on the carriageway.',
		start: 'Obstacle appears',
		end: 'Obstacle is removed',
	},
	fire_smoke: {
		title: 'Fire or smoke',
		family: 'flow',
		definition: 'Visible fire or smoke from a vehicle or on the road.',
		start: 'First visible smoke',
		end: 'Smoke clears or the frame ends',
	},
}

/** Tailwind classes using the --family-* tokens from globals.css. */
export const familyStyles: Record<EventFamily, { bar: string; dot: string }> = {
	pedestrian: { bar: 'bg-family-pedestrian', dot: 'bg-family-pedestrian' },
	collision: { bar: 'bg-family-collision', dot: 'bg-family-collision' },
	violation: { bar: 'bg-family-violation', dot: 'bg-family-violation' },
	flow: { bar: 'bg-family-flow', dot: 'bg-family-flow' },
}

export function classStyle(label: EventClass) {
	return familyStyles[classMeta[label].family]
}

/** Classes present in the events, in the official CLASSES order. */
export function presentClasses(events: TrafficEvent[]): EventClass[] {
	const present = new Set(events.map((event) => event[2]))
	return EVENT_CLASSES.filter((label) => present.has(label))
}

export function sortEvents(events: TrafficEvent[]) {
	return [...events].sort((a, b) => a[0] - b[0] || a[1] - b[1])
}
