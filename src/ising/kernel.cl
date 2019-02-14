// opencl kernel
// monte-carlo simulation

float rand(float x, float y, float z) {
    float ptr = 0.0f;
    return fract(sin(x * 112.9898f + y * 179.233f + z * 237.212f) * 43758.5453f, &ptr);
}

int dE(int s, int x, int y, int w, int h, local int * local_board) {
	const int ds = local_board[y * w + (x - 1)] +
				    local_board[(y - 1) * w + x] +
				    local_board[y * w + (x + 1)] +
				    local_board[(y + 1) * w + x];
	return ((- s * ds) / 2) + 2;
}

kernel void rng(global int * board, float seed) {
    const int X = get_global_id(0) + 1;
    const int Y = get_global_id(1) + 1;
    const int W = get_global_size(0) + 2;
    const int H = get_global_size(1) + 2;

	const float r = rand((float)Y / H, (float)X / W, seed);

	board[Y * W + X] = (r > 0.5) ? 1 : -1;
}

kernel void monte_carlo_step(global int * board,
							 local int * local_board,
							 float2 probabilities,
							 float seed,
							 volatile global int * out)
{
    const int X = get_global_id(0) + 1;
    const int Y = get_global_id(1) + 1;
    const int W = get_global_size(0) + 2;
    const int H = get_global_size(1) + 2;
    const int x = get_local_id(0) + 1;
    const int y = get_local_id(1) + 1;
    const int w = get_local_size(0) + 2;
    const int h = get_local_size(1) + 2;
    const int i = get_group_id(0);
    const int j = get_group_id(1);
	const int XY = X + Y;
	const int xy = x + y;
	const int ij = i + j;

	volatile local int local_out[3];
	local_out[0] = local_out[1] = local_out[2] = 0;

	local_board[y * w + x] = board[Y * W + X];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll
	for (int itb = 0; itb < 2; ++itb) {
		if ((ij & 1) == itb) {

			if (x == 1) local_board[y * w + 0] = board[Y * W + (X - 1)];
			else if (x + 2 == w) local_board[y * w + (x + 1)] = board[Y * W + (X + 1)];

			if (y == 1) local_board[0 * w + x] = board[(Y - 1) * W + X];
			else if (y + 2 == h) local_board[(y + 1) * w + x] = board[(Y + 1) * W + X];

			#pragma unroll
			for (int itt = 0; itt < 2; ++itt) {
				if ((xy & 1) == itt) {
					int s = local_board[y * w + x];
					int de = dE(s, x, y, w, h, local_board);
					
					const float r = rand((float)Y / H, (float)X / W, seed);

					if (de >= 2) {
						s = -s;
					} else if (r < probabilities[de]) {
						s = -s;
					}

					local_board[y * w + x] = s;
				}
	
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if (x == 1 || x + 2 == w || y == 1 || y + 2 == h)
				board[Y * W + X] = local_board[y * w + x];

			if (X == 1) board[Y * W + 0] = board[Y * W + (W - 2)];
			else if (X + 2 == W) board[Y * W + (X + 1)] = board[Y * W + 1];

			if (Y == 1) board[0 * W + X] = board[(H - 2) * W + X];
			else if (Y + 2 == H) board[(Y + 1) * W + X] = board[1 * W + X];
		}
	
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	board[Y * W + X] = local_board[y * w + x];

	if (local_board[y * w + x] == 1) atomic_inc(local_out);
	else							 atomic_inc(local_out + 1);
	
	atomic_add(local_out + 2, local_board[y * w + x] * (local_board[(y - 1) * w + x]) + local_board[y * w + (x - 1)]);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (x == 1 && y == 1) {
		atomic_add(out, local_out[0]);
		atomic_add(out + 1, local_out[1]);
		atomic_add(out + 2, local_out[2]);
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}
