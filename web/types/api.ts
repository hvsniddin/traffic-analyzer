export type ApiSuccess<TData> = {
	ok: true
	status: number
	data: TData
}

export type ApiError = {
	ok: false
	/** 0 when the server could not be reached. */
	status: number
	message: string
}

export type ApiResult<TData> = ApiSuccess<TData> | ApiError
