#pragma once

#include <cmath>
#include <utility>
#include <atomic>
#include <array>

namespace omp_kernel {

inline float rand(float x, float y, float z) {
    float ptr = 0.0f;
    float f = modf(sin(x * 112.9898f + y * 179.233f + z * 237.212f) * 43758.5453f, &ptr);
    return abs(f);
}

inline int dE(int s, int x, int y, int w, int h, int * nb) {
	const int ds = nb[0] + nb[1] + nb[2] + nb[3];
	return ((- s * ds) / 2) + 2;
}

inline void monte_carlo_step(size_t W, size_t H,
                             int * board,
					         std::array < float, 2 > probabilities,
					         float seed,
					         std::array < int, 3 > & out)
{
    for (int itb = 0; itb < 2; ++itb)
    for (int itt = 0; itt < 2; ++itt)
    {
        int np = 0, nm = 0, ss = 0;
    
        for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
        {
            if (((i + j) & 1) != itb) continue;

            size_t X0 = (i == 0) ? 1 : W / 2;
            size_t X1 = (i == 0) ? W / 2 : W - 1;
            size_t Y0 = (j == 0) ? 1 : H / 2;
            size_t Y1 = (j == 0) ? H / 2 : H - 1;

            #pragma omp parallel for firstprivate(itt) reduction(+:np,nm,ss)
            for (int X = X0; X < X1; ++X)
            for (int Y = Y0; Y < Y1; ++Y)
            {
                size_t XY = X + Y;
                size_t ij = ((X > W / 2) ? 1 : 0) + ((Y > H / 2) ? 1 : 0);
    
		        if ((XY & 1) == itt)
                {
	                int neighbors[4];
	                int s = 0;
    
			        s = board[Y * W + X];
    
			        neighbors[0] = board[Y * W + (X - 1)];
			        neighbors[1] = board[(Y - 1) * W + X];
			        neighbors[2] = board[Y * W + (X + 1)];
			        neighbors[3] = board[(Y + 1) * W + X];
    
			        int de = dE(s, X, Y, W, H, neighbors);
					
			        const float r = rand((float)Y / H, (float)X / W, seed);
    
			        if (de >= 2) {
				        s = -s;
			        } else if (r < probabilities[de]) {
				        s = -s;
			        }
    
			        board[Y * W + X] = s;
    
                    np += (s == 1) ? 1 : 0;
                    nm += (s == -1) ? 1 : 0;
                    ss += s * (neighbors[0] + neighbors[1]);
    
		            if (X == 1) board[Y * W + 0] = board[Y * W + (W - 2)];
		            else if (X + 2 == W) board[Y * W + (X + 1)] = board[Y * W + 1];
    
		            if (Y == 1) board[0 * W + X] = board[(H - 2) * W + X];
		            else if (Y + 2 == H) board[(Y + 1) * W + X] = board[1 * W + X];
		        }
	        }
    
            out[0] += np;
            out[1] += nm;
            out[2] += ss;
        }
    }
}

}