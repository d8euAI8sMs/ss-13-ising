#pragma once

#include <util/common/geom/point.h>
#include <util/common/math/vec.h>
#include <util/common/plot/plot.h>
#include <util/common/math/fuzzy.h>

#include <vector>
#include <map>
#include <array>

#include <fstream>
#include <streambuf>

#include <omp.h>
#include <CL/cl2.hpp>

#include "aligned_allocator.h"
#include "openmp_kernel.h"

namespace model
{

    template < typename T >
    using vector4096 = std::vector < T, aligned_allocator < T, 4096 > > ;

    /*****************************************************/
    /*                     params                        */
    /*****************************************************/

    namespace consts
    {
        static const double k = 8.617e-5; /* eV / K */
    };

    struct parameters
    {
        // system params
        size_t n;

        // other params
        double J /* eV */;

        // computation params
        bool opencl_gpu;

        double Tc() const { return 2.269 * std::abs(J) / consts::k; }
    };

    inline parameters make_default_parameters()
    {
        parameters p =
        {
            // system params
            16,

            // other params
            1,

            // computation params
            true,
        };
        return p;
    }

    /*****************************************************/
    /*                    opencl                         */
    /*****************************************************/

    struct opencl_data
    {
        cl::Platform platform;
        cl::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;
        cl::Kernel kernel;
        cl::Buffer buffer;
        cl::Buffer out_buffer;
        vector4096 < cl_int > out;
        size_t board_w, w;
        size_t max_w;
    };

    inline void init_opencl_common(opencl_data & d, bool gpu = true)
    {
        d.platform = cl::Platform::getDefault();
    
        d.platform.getDevices(gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, &d.devices);
    
        if (d.devices.empty()) throw "no valid cpu/gpu device found";
    
        d.context = cl::Context(d.devices[0]);
    
        d.queue = cl::CommandQueue(d.context, d.devices[0]);

        std::ifstream input;
        input.open("kernel.cl");
        cl::string src((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());

        d.program = cl::Program(d.context, src);
        d.program.build("-cl-fast-relaxed-math "
                        "-cl-finite-math-only "
                        "-cl-unsafe-math-optimizations "
                        "-cl-no-signed-zeros "
                        "-cl-mad-enable");

        d.kernel = cl::Kernel(d.program, "monte_carlo_step");

        d.max_w = min(16, std::sqrt(d.devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()));
        if (gpu) d.max_w /= 2;
    }

    inline void init_opencl_data(opencl_data & d, vector4096<cl_int> & data, size_t bw)
    {
        d.board_w = bw;
        d.w = min(bw, d.max_w);
        d.buffer = cl::Buffer(d.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                              data.get_allocator().optimal_size(data.size()), (void*)data.data());
        d.kernel.setArg(0, d.buffer);
        d.kernel.setArg(1, (d.w + 2) * (d.w + 2) * sizeof(cl_int), NULL);
    }

    inline void opencl_exec(opencl_data & d, cl_float2 probs, size_t batch_size)
    {
        d.out.clear();
        d.out.resize(3 * batch_size);
        d.out_buffer = cl::Buffer(d.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                  d.out.get_allocator().optimal_size(d.out.size()), (void*)d.out.data());
        d.kernel.setArg(5, d.out_buffer);

        d.kernel.setArg(4, (int)batch_size);

        d.kernel.setArg(2, probs);

        d.kernel.setArg(3, rand() / (RAND_MAX + 1.f));

        d.queue.enqueueNDRangeKernel(d.kernel, cl::NullRange, { d.board_w, d.board_w }, { d.w, d.w });

        d.queue.finish();
    }

    /*****************************************************/
    /*                     data                          */
    /*****************************************************/

    class board
    {
    public:
        struct macroparams
        {
            double m, e, c, hi;
        };
        struct avgparams
        {
            double ea, ma, e2a, m2a;
            size_t n;
            const parameters * p;
            double T;
            double w[2];
        };
    public:
        size_t w;
        vector4096 < int > data;
        macroparams params;
        opencl_data * ocl_data;
    private:
        avgparams aparams;
    public:
        void init(const parameters & p)
        {
            w = p.n;
            data.clear();
            data.resize((w + 2) * (w + 2));
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] = ((rand() & 1) == 1) ? 1 : -1;
            }
            ensure_periodic(p);
            aparams = { 0, 0, 0, 0, 0, nullptr, 0 };
            
            init_opencl_common(*ocl_data, p.opencl_gpu);
            init_opencl_data(*ocl_data, data, w);
        }

        void begin(const parameters & p, double T)
        {
            aparams = { 0, 0, 0, 0, 0, &p, T };
            aparams.w[0] = std::exp(- 8 * std::abs(p.J) / consts::k / T);
            aparams.w[1] = std::exp(- 4 * std::abs(p.J) / consts::k / T);
        }

        void next(bool opencl, size_t batch_size)
        {
            if (opencl) next_opencl(batch_size);
            else        next_openmp(batch_size);
        }

        void next_opencl(size_t batch_size) {
            opencl_exec(*ocl_data, cl_float2 { aparams.w[0], aparams.w[1] }, batch_size);

            for (size_t i = 0; i < batch_size; ++i)
            {
                params.m = std::abs((double)ocl_data->out[0 + i * 3] - (double)ocl_data->out[1 + i * 3]) / w / w / w / w;

                params.e = std::abs(aparams.p->J * ocl_data->out[2 + i * 3] / w / w);

                aparams.ea += params.e;
                aparams.ma += params.m;

                aparams.e2a += params.e * params.e;
                aparams.m2a += params.m * params.m;

                ++aparams.n;
            }
        }

        void next_openmp(size_t batch_size) {
            for (size_t i = 0; i < batch_size; ++i) next_openmp();
        }

        void next_openmp() {
            std::array < int, 3 > out = {{ 0, 0, 0 }};

            omp_kernel::monte_carlo_step(w + 2, w + 2, data.data(), {{ (float)aparams.w[0], (float)aparams.w[1] }}, rand() / (RAND_MAX + 1.f), out);

            params.m = std::abs((double)out[0] - (double)out[1]) / w / w / w / w;

            params.e = std::abs(aparams.p->J * out[2] / w / w);

            aparams.ea += params.e;
            aparams.ma += params.m;

            aparams.e2a += params.e * params.e;
            aparams.m2a += params.m * params.m;

            ++aparams.n;
        }

        void end()
        {
            const size_t n = aparams.p->n;

            aparams.e2a /= aparams.n;
            aparams.ea /= aparams.n;

            aparams.m2a /= aparams.n;
            aparams.ma /= aparams.n;

            params.e = aparams.ea;
            params.m = aparams.ma;
            params.c = n * n / (consts::k * aparams.T) / (consts::k * aparams.T) * std::abs(aparams.e2a - aparams.ea * aparams.ea);
            params.hi = n * n / (consts::k * aparams.T) * std::abs(aparams.m2a - aparams.ma * aparams.ma);
        }

    private:

        int spin_at(size_t i, size_t j) const { return data[i * (w + 2) + j]; }

        void ensure_periodic(const parameters & p)
        {
            #pragma omp parallel for
            for (int i = 0; i < p.n + 2; ++i)
            {
                data[i * (p.n + 2) + 0] = data[i * (p.n + 2) + p.n];
                data[i * (p.n + 2) + p.n + 1] = data[i * (p.n + 2) + 1];
                data[0 * (p.n + 2) + i] = data[p.n * (p.n + 2) + i];
                data[(p.n + 1) * (p.n + 2) + i] = data[1 * (p.n + 2) + i];
            }
        }
    };

    /*****************************************************/
    /*                     drawing                       */
    /*****************************************************/

    using points_t = std::vector < geom::point2d_t > ;

    struct plot_data
    {
        util::ptr_t < points_t > data;
        plot::list_drawable < points_t > :: ptr_t plot;
        plot::world_t::ptr_t world;
        plot::auto_viewport < points_t > :: ptr_t autoworld;
    };

    struct model_data
    {
        util::ptr_t < parameters > params;
        plot_data   e_data;
        plot_data   m_data;
        plot_data   c_data;
        plot_data   hi_data;
        util::ptr_t < opencl_data > ocl_data;
        board       system_data;
    };

    inline static plot_data make_plot_data
    (
        plot::palette::pen_ptr pen = plot::palette::pen(0xffffff),
        plot::list_data_format data_format = plot::list_data_format::chain
    )
    {
        plot_data pd;
        pd.world = plot::world_t::create();
        pd.autoworld = plot::min_max_auto_viewport < points_t > :: create();
        pd.data = util::create < points_t > ();
        pd.plot = plot::list_drawable < points_t > :: create
        (
            plot::make_data_source(pd.data),
            nullptr, // no point painter
            pen
        );
        pd.plot->data_format = data_format;
        return pd;
    }

    inline static plot::drawable::ptr_t make_root_drawable
    (
        const plot_data & p,
        std::vector < plot::drawable::ptr_t > layers
    )
    {
        using namespace plot;

        return viewporter::create(
            tick_drawable::create(
                layer_drawable::create(layers),
                const_n_tick_factory<axe::x>::create(
                    make_simple_tick_formatter(6, 8),
                    0,
                    5
                ),
                const_n_tick_factory<axe::y>::create(
                    make_simple_tick_formatter(6, 8),
                    0,
                    5
                ),
                palette::pen(RGB(80, 80, 80)),
                RGB(200, 200, 200)
            ),
            make_viewport_mapper(make_world_mapper < points_t > (p.autoworld))
        );
    }

    inline std::unique_ptr < CBitmap > export_system(const board & b, bool borders)
    {
        if (b.data.empty()) return {};

        CDC dc; dc.CreateCompatibleDC(nullptr);

        size_t cw = 5, ch = 5;

        auto bmp = std::make_unique < CBitmap > ();
        bmp->CreateBitmap(cw * b.w, ch * b.w, 1, 32, NULL);
        bmp->SetBitmapDimension(cw * b.w, ch * b.w);

        dc.SelectObject(bmp.get());

        auto nbrush = plot::palette::brush(RGB(150,0,0));
        auto pbrush = plot::palette::brush(RGB(0,150,0));

        CRect r;

        for (size_t i = 0; i < b.w; ++i)
        for (size_t j = 0; j < b.w; ++j)
        {
            bool v = b.data[(i + 1) * (b.w + 2) + j + 1] == 1;
            r.left = cw * i; r.right = cw * (i + 1);
            r.top = ch * j; r.bottom = ch * (j + 1);
            if (borders)
            {
                dc.SelectObject(v ? pbrush.get() : nbrush.get());
                dc.Rectangle(cw * i,
                             ch * j,
                             cw * (i + 1),
                             ch * (j + 1));
            }
            else
            {
                dc.FillRect(&r, v ? pbrush.get() : nbrush.get());
            }
        }

        dc.SelectObject((CBitmap *) nullptr);

        return bmp;
    }

    inline plot::drawable::ptr_t make_system_plot(const board & b)
    {
        return plot::custom_drawable::create([&b] (CDC & dc, const plot::viewport & vp)
        {
            if (b.data.empty()) return;

            size_t cw = vp.screen.width() / b.w;
            size_t ch = vp.screen.height() / b.w;

            auto nbrush = plot::palette::brush(RGB(150,0,0));
            auto pbrush = plot::palette::brush(RGB(0,150,0));

            if (cw > 2 && ch > 2)
            {
                CRect r;
                for (size_t i = 0; i < b.w; ++i)
                for (size_t j = 0; j < b.w; ++j)
                {
                    bool v = b.data[(i + 1) * (b.w + 2) + j + 1] == 1;
                    dc.SelectObject(v ? pbrush.get() : nbrush.get());
                    dc.Rectangle(vp.screen.xmin + cw * i,
                                 vp.screen.ymin + ch * j,
                                 vp.screen.xmin + cw * (i + 1),
                                 vp.screen.ymin + ch * (j + 1));
                }
            }
            else
            {
                auto bmp = export_system(b, false);
                CDC memDC; memDC.CreateCompatibleDC(&dc);
                memDC.SelectObject(bmp.get());
                dc.StretchBlt(vp.screen.xmin, vp.screen.ymin, vp.screen.width(), vp.screen.height(), &memDC, 0, 0, bmp->GetBitmapDimension().cx, bmp->GetBitmapDimension().cy, SRCCOPY);
                memDC.SelectObject((CBitmap *) nullptr);
            }
        });
    }

    inline model_data make_model_data(const parameters & p = make_default_parameters())
    {
        model_data md;
        md.params = util::create < parameters > (p);
        md.e_data = make_plot_data(plot::palette::pen(0x0000ff, 2));
        md.m_data = make_plot_data(plot::palette::pen(0x0000ff, 2));
        md.c_data = make_plot_data(plot::palette::pen(0x0000ff, 2));
        md.hi_data = make_plot_data(plot::palette::pen(0x0000ff, 2));
        md.ocl_data = util::create < opencl_data > ();
        md.system_data.ocl_data = md.ocl_data.get();
        return md;
    }
}