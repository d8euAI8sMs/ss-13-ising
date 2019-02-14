#pragma once

#include <util/common/geom/point.h>
#include <util/common/math/vec.h>
#include <util/common/plot/plot.h>
#include <util/common/math/fuzzy.h>

#include <vector>
#include <map>
#include <array>

#include <omp.h>

namespace model
{

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

        double Tc() const { return 2.269 * std::abs(J) / consts::k; }
    };

    inline parameters make_default_parameters()
    {
        parameters p =
        {
            // system params
            10,

            // other params
            1
        };
        return p;
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
        std::vector < int > data;
        macroparams params;
    private:
        avgparams aparams;
    public:
        void init(const parameters & p)
        {
            w = p.n;
            data.clear();
            data.resize((w + 2) * (w * 2));
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] = ((rand() & 1) == 1) ? 1 : -1;
            }
            ensure_periodic(p);
            aparams = { 0, 0, 0, 0, 0, nullptr, 0 };
        }

        void begin(const parameters & p, double T)
        {
            aparams = { 0, 0, 0, 0, 0, &p, T };
            aparams.w[0] = std::exp(- 8 * std::abs(p.J) / consts::k / T);
            aparams.w[1] = std::exp(- 4 * std::abs(p.J) / consts::k / T);
        }

        void next()
        {
            int s = 0;

            if ((w - 2) < 4 * omp_get_max_threads())
                next_linear(w);
            else
                next_parallel(w);
            
            if (aparams.p->J > 0)
            {
                size_t np = 0, nm = 0;
                for (size_t i = 1; i < w + 1; ++i)
                #pragma omp parallel for reduction(+:np,nm,s) firstprivate(i)
                for (int j = 1; j < w + 1; ++j)
                {
                    np += data[i * (w + 2) + j] == 1 ? 1 : 0;
                    nm += data[i * (w + 2) + j] == -1 ? 0 : 1;
                    s += spin_at(i, j) * (spin_at(i - 1, j) + spin_at(i, j - 1));
                }
                params.m = std::abs((double)np - (double)nm) / w / w / w / w;
            }
            else
            {
                size_t n1p = 0, n1m = 0, n2p = 0, n2m = 0;
                for (size_t i = 1; i < w + 1; ++i)
                {
                    #pragma omp parallel for reduction(+:n1p,n1m,n2p,n2m,s) firstprivate(i)
                    for (int j = 1 + (i & 1); j < w + 1; j += 2)
                    {
                        n1p += data[i * (w + 2) + j] == 1 ? 1 : 0;
                        n1m += data[i * (w + 2) + j] == -1 ? 0 : 1;
                        n2p += data[i * (w + 2) + j + 1] == 1 ? 1 : 0;
                        n2m += data[i * (w + 2) + j + 1] == -1 ? 0 : 1;
                        s += spin_at(i, j) * (spin_at(i - 1, j) + spin_at(i, j - 1));
                    }
                    #pragma omp parallel for reduction(+:n1p,n1m,n2p,n2m,s) firstprivate(i)
                    for (int j = 1 + (1 - (i & 1)); j < w + 1; j += 2)
                    {
                        n1p += data[i * (w + 2) + j + 1] == 1 ? 1 : 0;
                        n1m += data[i * (w + 2) + j + 1] == -1 ? 0 : 1;
                        n2p += data[i * (w + 2) + j] == 1 ? 1 : 0;
                        n2m += data[i * (w + 2) + j] == -1 ? 0 : 1;
                        s += spin_at(i, j) * (spin_at(i - 1, j) + spin_at(i, j - 1));
                    }
                }
                params.m = std::abs(((double)n1p - (double)n1m) - ((double)n2p - (double)n2m)) / w / w / w / w;
            }

            params.e = std::abs(aparams.p->J * s / w / w);

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

        void ensure_periodic(const parameters & p, size_t i, size_t j)
        {
            data[(p.n + 1) * (p.n + 2) + j] = data[1 * (p.n + 2) + j];
            data[i * (p.n + 2) + p.n + 1] = data[i * (p.n + 2) + 1];
            data[0 * (p.n + 2) + j] = data[p.n * (p.n + 2) + j];
            data[i * (p.n + 2) + 0] = data[i * (p.n + 2) + p.n];
        }

        size_t dE(size_t i, size_t j) const
        {
            int s = spin_at(i, j - 1) + spin_at(i, j + 1) +
                    spin_at(i - 1, j) + spin_at(i + 1, j);
            s *= aparams.p->J < 0 ? -1 : 1;
            return - spin_at(i, j) * s / 2 + 2;
        }

        void next_linear(size_t n)
        {
            for (size_t l = 0; l < n * n; ++l)
            {
                size_t i = (rand() % n) + 1;
                size_t j = (rand() % n) + 1;
                size_t c = dE(i, j);
                if (c >= 2) data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                else
                {
                    if (rand() < aparams.w[c] * RAND_MAX)
                        data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                }
                ensure_periodic(*aparams.p, i, j);
            }
        }

        void next_parallel(size_t n)
        {
            std::vector < std::array < size_t, 3 > > ranges(omp_get_max_threads());
            size_t d = (size_t) std::ceil((double) n / omp_get_max_threads());
            for (size_t i = 0; (i < omp_get_max_threads()) && (n > i * d); ++i)
            {
                ranges[i][0] = i * d;
                ranges[i][1] = min((i + 1) * d, n);
                ranges[i][2] = min(d, n - i * d) * n;
            }

            #pragma omp parallel firstprivate(ranges)
            {
                auto tid = omp_get_thread_num();
                auto b = std::get<0>(ranges[tid]);
                auto e = std::get<1>(ranges[tid]);
                for (size_t l = 0; l < std::get<2>(ranges[tid]); ++l)
                {
                    size_t i = b + rand() % (e - b) + 1;
                    size_t j = (rand() % n) + 1;
                    if (i == 1 || i == e || j == 1 || j == n)
                    {
                        #pragma omp critical
                        {
                            size_t c = dE(i, j);
                            if (c >= 2) data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                            else
                            {
                                if (rand() < aparams.w[c] * RAND_MAX)
                                    data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                            }
                            ensure_periodic(*aparams.p, i, j);
                        }
                    }
                    else
                    {
                        size_t c = dE(i, j);
                        if (c >= 2) data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                        else
                        {
                            if (rand() < aparams.w[c] * RAND_MAX)
                                data[i * (n + 2) + j] = -data[i * (n + 2) + j];
                        }
                        if (i == n || i == 1 || j == n || j == 1)
                        {
                            #pragma omp critical
                            ensure_periodic(*aparams.p, i, j);
                        }
                    }
                }
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
        return md;
    }
}