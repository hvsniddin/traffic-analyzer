import { z } from 'zod'

import { EVENT_CLASSES } from './events'

// Contract from docs/demo_plan.md ("API contract for parallel work").
export const jobStatuses = ['queued', 'running', 'done', 'error'] as const

export type JobStatus = (typeof jobStatuses)[number]

const eventSchema = z.tuple([z.number(), z.number(), z.enum(EVENT_CLASSES)])
const riskPointSchema = z.tuple([z.number(), z.number()])

// [x1, y1, x2, y2 (0-1), detector class id, track id, event labels]
const overlayObjectSchema = z.tuple([
	z.number(), z.number(), z.number(), z.number(), z.number(), z.number(), z.array(z.string()),
])

export const overlaySchema = z.object({
	scene: z.array(z.object({ kind: z.string(), name: z.string(), points: z.array(z.tuple([z.number(), z.number()])) })),
	frames: z.array(z.object({ t: z.number(), objects: z.array(overlayObjectSchema) })),
})

export const jobResultSchema = z.object({
	duration_sec: z.number(),
	events: z.array(eventSchema),
	// null until Part B is genuinely implemented.
	risk: z.array(riskPointSchema).nullable(),
	// Optional compressed annotated clip; the timeline works without it.
	annotated_video_url: z.string().nullish(),
	// Per sampled frame boxes and aligned scene geometry, drawn over the video.
	overlay: overlaySchema.nullish(),
})

export const jobStateSchema = z.object({
	job_id: z.string(),
	status: z.enum(jobStatuses),
	stage: z.string().nullish(),
	// 0–1 when measured, otherwise null. Never shown as a percentage when null.
	progress: z.number().min(0).max(1).nullable(),
	result: jobResultSchema.nullable(),
	error: z.string().nullable(),
})

export const jobCreatedSchema = z.object({ job_id: z.string() })

export type JobResult = z.infer<typeof jobResultSchema>
export type Overlay = z.infer<typeof overlaySchema>
export type JobState = z.infer<typeof jobStateSchema>
export type JobCreated = z.infer<typeof jobCreatedSchema>
