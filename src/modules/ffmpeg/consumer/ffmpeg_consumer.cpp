/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#include "ffmpeg_consumer.h"

#include "../util/av_assert.h"
#include "../util/av_util.h"

#include <common/bit_depth.h>
#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/executor.h>
#include <common/future.h>
#include <common/memory.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>
#include <core/video_format.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4245)
#pragma warning(disable : 4701)
#include <boost/crc.hpp>
#pragma warning(pop)

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <tbb/concurrent_queue.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <future>
#include <memory>
#include <optional>
#include <thread>

namespace caspar { namespace ffmpeg {

// TODO multiple output streams
// TODO multiple output files
// TODO run video filter, video encoder, audio filter, audio encoder in separate threads.
// TODO realtime with smaller buffer?

struct Stream
{
    std::shared_ptr<AVFilterGraph> graph  = nullptr;
    AVFilterContext*               sink   = nullptr;
    AVFilterContext*               source = nullptr;

    std::shared_ptr<AVCodecContext> enc = nullptr;
    AVStream*                       st  = nullptr;

    Stream(AVFormatContext*                    oc,
           std::string                         suffix,
           AVCodecID                           codec_id,
           const core::video_format_desc&      format_desc,
           bool                                realtime,
           common::bit_depth                   depth,
           std::map<std::string, std::string>& options)
    {
        std::map<std::string, std::string> stream_options;

        {
            auto tmp = std::move(options);
            for (auto& p : tmp) {
                if (boost::algorithm::ends_with(p.first, suffix)) {
                    const auto key = p.first.substr(0, p.first.size() - suffix.size());
                    stream_options.emplace(key, std::move(p.second));
                } else {
                    options.insert(std::move(p));
                }
            }
        }

        std::string filter_spec = "";
        {
            const auto it = stream_options.find("filter");
            if (it != stream_options.end()) {
                filter_spec = std::move(it->second);
                stream_options.erase(it);
            }
        }

        auto codec = avcodec_find_encoder(codec_id);
        {
            const auto it = stream_options.find("codec");
            if (it != stream_options.end()) {
                codec = avcodec_find_encoder_by_name(it->second.c_str());
                stream_options.erase(it);
            }
        }

        if (!codec) {
            FF_RET(AVERROR(EINVAL), "avcodec_find_encoder");
        }

        AVFilterInOut* outputs = nullptr;
        AVFilterInOut* inputs  = nullptr;

        CASPAR_SCOPE_EXIT
        {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
        };

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }
        } else {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }
        }

        FF(avfilter_graph_parse2(graph.get(), filter_spec.c_str(), &inputs, &outputs));

        {
            auto cur = inputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph input count"));
            }

            if (codec->type == AVMEDIA_TYPE_VIDEO) {
                const auto sar = boost::rational<int>(format_desc.square_width, format_desc.square_height) /
                                 boost::rational<int>(format_desc.width, format_desc.height);

                const auto pix_fmt = (depth == common::bit_depth::bit8) ? AV_PIX_FMT_BGRA : AV_PIX_FMT_BGRA64LE;

                auto args = (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:sar=%d/%d:frame_rate=%d/%d") %
                             format_desc.width % format_desc.height % pix_fmt % format_desc.duration %
                             (format_desc.time_scale * format_desc.field_count) % sar.numerator() % sar.denominator() %
                             (format_desc.framerate.numerator() * format_desc.field_count) %
                             format_desc.framerate.denominator())
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &source, avfilter_get_by_name("buffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
                auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") % 1 %
                             format_desc.audio_sample_rate % format_desc.audio_sample_rate % AV_SAMPLE_FMT_S32 %
                             get_channel_layout_mask_for_channels(format_desc.audio_channels))
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter input media type"));
            }
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, graph.get()));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            // TODO codec->profiles
            // TODO FF(av_opt_set_int_list(sink, "framerates", codec->supported_framerates, { 0, 0 },
            // AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "pix_fmts", codec->pix_fmts, -1, AV_OPT_SEARCH_CHILDREN));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            FF(avfilter_graph_create_filter(
                &sink, avfilter_get_by_name("abuffersink"), "out", nullptr, nullptr, graph.get()));
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#endif
            // TODO codec->profiles
            FF(av_opt_set_int_list(sink, "sample_fmts", codec->sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "sample_rates", codec->supported_samplerates, 0, AV_OPT_SEARCH_CHILDREN));

            // TODO: need to translate codec->ch_layouts into something that can be passed via av_opt_set_*
            // FF(av_opt_set_chlayout(sink, "ch_layouts", codec->ch_layouts, AV_OPT_SEARCH_CHILDREN));

#ifdef _MSC_VER
#pragma warning(pop)
#endif
        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != codec->type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        st = avformat_new_stream(oc, nullptr);
        if (!st) {
            FF_RET(AVERROR(ENOMEM), "avformat_new_stream");
        }

        enc = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

        if (!enc) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3")
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            st->time_base = av_inv_q(av_buffersink_get_frame_rate(sink));

            // Ensure the frame_rate is set in a way that rtmp will find it
            st->avg_frame_rate = av_buffersink_get_frame_rate(sink);

            enc->width               = av_buffersink_get_w(sink);
            enc->height              = av_buffersink_get_h(sink);
            enc->framerate           = av_buffersink_get_frame_rate(sink);
            enc->sample_aspect_ratio = av_buffersink_get_sample_aspect_ratio(sink);
            enc->time_base           = st->time_base;
            enc->pix_fmt             = static_cast<AVPixelFormat>(av_buffersink_get_format(sink));
        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            st->time_base = {1, av_buffersink_get_sample_rate(sink)};

            enc->sample_fmt  = static_cast<AVSampleFormat>(av_buffersink_get_format(sink));
            enc->sample_rate = av_buffersink_get_sample_rate(sink);
            enc->time_base   = st->time_base;

            FF(av_buffersink_get_ch_layout(sink, &enc->ch_layout));

        } else {
            // TODO
        }

        if (realtime && codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
            enc->thread_type = FF_THREAD_SLICE;
        }

        if (oc->oformat->flags & AVFMT_GLOBALHEADER) {
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        auto dict = to_dict(std::move(stream_options));
        CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
        FF(avcodec_open2(enc.get(), codec, &dict));
        for (auto& p : to_map(&dict)) {
            options[p.first] = p.second + suffix;
        }

        FF(avcodec_parameters_from_context(st->codecpar, enc.get()));

        if (codec->type == AVMEDIA_TYPE_AUDIO && !(codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) {
            av_buffersink_set_frame_size(sink, enc->frame_size);
        }
    }

    void send(std::tuple<core::const_frame, std::int64_t, std::int64_t>& data,
              const core::video_format_desc&                             format_desc,
              std::function<void(std::shared_ptr<AVPacket>)>             cb)
    {
        std::shared_ptr<AVFrame>  frame;
        std::shared_ptr<AVPacket> pkt;

        const auto [in_frame, video_pts, audio_pts] = data;

        if (in_frame) {
            if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {
                frame      = make_av_video_frame(in_frame, format_desc);
                frame->pts = video_pts;
            } else if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {
                frame      = make_av_audio_frame(in_frame, format_desc);
                frame->pts = audio_pts;
            } else {
                // TODO
            }
            FF(av_buffersrc_write_frame(source, frame.get()));
        } else {
            FF(av_buffersrc_close(source, AV_NOPTS_VALUE, 0));
        }

        while (true) {
            pkt     = alloc_packet();
            int ret = avcodec_receive_packet(enc.get(), pkt.get());

            if (ret == AVERROR(EAGAIN)) {
                frame = alloc_frame();
                ret   = av_buffersink_get_frame(sink, frame.get());
                if (ret == AVERROR(EAGAIN)) {
                    return;
                }
                if (ret == AVERROR_EOF) {
                    FF(avcodec_send_frame(enc.get(), nullptr));
                } else {
                    FF_RET(ret, "av_buffersink_get_frame");
                    FF(avcodec_send_frame(enc.get(), frame.get()));
                }
            } else if (ret == AVERROR_EOF) {
                return;
            } else {
                FF_RET(ret, "avcodec_receive_packet");
                pkt->stream_index = st->index;
                av_packet_rescale_ts(pkt.get(), enc->time_base, st->time_base);
                cb(std::move(pkt));
            }
        }
    }
};

struct ffmpeg_consumer : public core::frame_consumer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;
    int                     channel_index_ = -1;
    core::video_format_desc format_desc_;
    bool                    realtime_ = false;
    std::int64_t            video_pts = 0;
    std::int64_t            audio_pts = 0;

    spl::shared_ptr<diagnostics::graph> graph_;

    std::string path_;
    std::string args_;

    std::exception_ptr exception_;
    std::mutex         exception_mutex_;
    std::atomic<bool>  failed_{false};

    tbb::concurrent_bounded_queue<std::tuple<core::const_frame, std::int64_t, std::int64_t>> frame_buffer_;
    std::thread                                                                              frame_thread_;

    common::bit_depth depth_;

  public:
    ffmpeg_consumer(std::string path, std::string args, bool realtime, common::bit_depth depth)
        : channel_index_([&] {
            boost::crc_16_type result;
            result.process_bytes(path.data(), path.length());
            return result.checksum();
        }())
        , realtime_(realtime)
        , path_(std::move(path))
        , args_(std::move(args))
        , depth_(depth)
    {
        state_["file/path"] = u8(path_);

        frame_buffer_.set_capacity(realtime_ ? 1 : 64);

        diagnostics::register_graph(graph_);
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("input", diagnostics::color(0.7f, 0.4f, 0.4f));
    }

    ~ffmpeg_consumer()
    {
        if (frame_thread_.joinable()) {
            frame_buffer_.push({core::const_frame{}, -1, -1});
            frame_thread_.join();
        }
    }

    // frame consumer

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        if (frame_thread_.joinable()) {
            CASPAR_THROW_EXCEPTION(invalid_operation() << msg_info("Cannot reinitialize ffmpeg-consumer."));
        }

        format_desc_   = format_desc;
        channel_index_ = channel_info.index;

        graph_->set_text(print());

        frame_thread_ = std::thread([=] {
            try {
                std::map<std::string, std::string> options;
                {
                    static boost::regex opt_exp("-(?<NAME>[^\\s]+)(\\s+(?<VALUE>[^\\s]+))?");
                    for (auto it = boost::sregex_iterator(args_.begin(), args_.end(), opt_exp);
                         it != boost::sregex_iterator();
                         ++it) {
                        options[(*it)["NAME"].str().c_str()] =
                            (*it)["VALUE"].matched ? (*it)["VALUE"].str().c_str() : "";
                    }
                }

                boost::filesystem::path full_path = path_;

                static boost::regex prot_exp("^.+:.*");
                if (!boost::regex_match(path_, prot_exp)) {
                    if (!full_path.is_absolute()) {
                        full_path = u8(env::media_folder()) + path_;
                    }

                    // TODO -y?
                    if (boost::filesystem::exists(full_path)) {
                        boost::filesystem::remove(full_path);
                    }

                    boost::filesystem::create_directories(full_path.parent_path());
                }

                AVFormatContext* oc = nullptr;

                {
                    std::string format;
                    {
                        const auto format_it = options.find("format");
                        if (format_it != options.end()) {
                            format = std::move(format_it->second);
                            options.erase(format_it);
                        }
                    }

                    FF(avformat_alloc_output_context2(
                        &oc, nullptr, !format.empty() ? format.c_str() : nullptr, path_.c_str()));
                }

                CASPAR_SCOPE_EXIT { avformat_free_context(oc); };

                std::optional<Stream> video_stream;
                if (oc->oformat->video_codec != AV_CODEC_ID_NONE) {
                    if (oc->oformat->video_codec == AV_CODEC_ID_H264 && options.find("preset:v") == options.end()) {
                        options["preset:v"] = "veryfast";
                    }
                    video_stream.emplace(oc, ":v", oc->oformat->video_codec, format_desc, realtime_, depth_, options);

                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/fps"] = av_q2d(av_buffersink_get_frame_rate(video_stream->sink));
                    }
                }

                std::optional<Stream> audio_stream;
                if (oc->oformat->audio_codec != AV_CODEC_ID_NONE) {
                    audio_stream.emplace(oc, ":a", oc->oformat->audio_codec, format_desc, realtime_, depth_, options);
                }

                if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                    // TODO (fix) interrupt_cb
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avio_open2(&oc->pb, full_path.string().c_str(), AVIO_FLAG_WRITE, nullptr, &dict));
                    options = to_map(&dict);
                }

                {
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avformat_write_header(oc, &dict));
                    options = to_map(&dict);
                }

                {
                    for (auto& p : options) {
                        CASPAR_LOG(warning) << print() << " Unused option " << p.first << "=" << p.second;
                    }
                }

                tbb::concurrent_bounded_queue<std::shared_ptr<AVPacket>> packet_buffer;
                packet_buffer.set_capacity(realtime_ ? 1 : 128);
                std::promise<void> packet_thread_promise;
                auto packet_thread_future = packet_thread_promise.get_future();
                
                auto packet_thread = std::thread([&, promise = std::move(packet_thread_promise)]() mutable {
                    try {
                        CASPAR_SCOPE_EXIT
                        {
                            if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                                try {
                                    FF(avio_closep(&oc->pb));
                                } catch (...) {
                                    CASPAR_LOG_CURRENT_EXCEPTION();
                                    failed_.store(true, std::memory_order_release);
                                    throw;
                                }
                            }
                        };

                        std::map<int, int64_t> count;

                        std::shared_ptr<AVPacket> pkt;
                        while (true) {
                            packet_buffer.pop(pkt);
                            if (!pkt) {
                                break;
                            }
                            count[pkt->stream_index] += 1;
                            FF(av_interleaved_write_frame(oc, pkt.get()));
                        }

                        auto video_st = video_stream ? video_stream->st : nullptr;
                        auto audio_st = audio_stream ? audio_stream->st : nullptr;

                        if ((!video_st || count[video_st->index]) && (!audio_st || count[audio_st->index])) {
                            FF(av_write_trailer(oc));
                        }
                        
                        promise.set_value();

                    } catch (...) {
                        CASPAR_LOG_CURRENT_EXCEPTION();
                        failed_.store(true, std::memory_order_release);
                        packet_buffer.abort();
                        try {
                            promise.set_exception(std::current_exception());
                        } catch (...) {
                            // Promise already set or moved
                        }
                    }
                });
                CASPAR_SCOPE_EXIT
                {
                    if (packet_thread.joinable()) {
                        // TODO Is nullptr needed?
                        packet_buffer.push(nullptr);
                        packet_buffer.abort();
                        packet_thread.join();
                        
                        // Check if packet thread had any errors
                        try {
                            packet_thread_future.get();
                        } catch (...) {
                            failed_.store(true, std::memory_order_release);
                            std::lock_guard<std::mutex> lock(exception_mutex_);
                            if (exception_ == nullptr) {
                                exception_ = std::current_exception();
                            }
                        }
                    }
                };

                auto packet_cb = [&](std::shared_ptr<AVPacket>&& pkt) { packet_buffer.push(std::move(pkt)); };

                std::int64_t frame_number = 0;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/frame"] = frame_number++;
                    }

                    std::tuple<core::const_frame, std::int64_t, std::int64_t> data;
                    frame_buffer_.pop(data);
                    graph_->set_value("input",
                                      static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

                    caspar::timer frame_timer;
                    tbb::parallel_invoke(
                        [&] {
                            if (video_stream) {
                                video_stream->send(data, format_desc, packet_cb);
                            }
                        },
                        [&] {
                            if (audio_stream) {
                                audio_stream->send(data, format_desc, packet_cb);
                            }
                        });
                    graph_->set_value("frame-time", frame_timer.elapsed() * format_desc.fps * 0.5);

                    if (!std::get<0>(data)) {
                        packet_buffer.push(nullptr);
                        break;
                    }
                }

                packet_thread.join();
            } catch (...) {
                CASPAR_LOG_CURRENT_EXCEPTION();
                failed_.store(true, std::memory_order_release);
                std::lock_guard<std::mutex> lock(exception_mutex_);
                exception_ = std::current_exception();
            }
        });
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        // TODO - field alignment

        // Check failed flag first (synchronous, fast)
        if (failed_.load(std::memory_order_acquire)) {
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("FFmpeg consumer has failed"));
        }

        // Also check exception_ for backwards compatibility
        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

        if (!frame_buffer_.try_push({frame, video_pts, audio_pts})) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
        }

        video_pts += 1;
        audio_pts += frame.audio_data().size() / format_desc_.audio_channels;

        graph_->set_value("input", static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

        return make_ready_future(true);
    }

    std::wstring print() const override { return L"ffmpeg[" + u16(path_) + L"]"; }

    std::wstring name() const override { return L"ffmpeg"; }

    bool has_synchronization_clock() const override { return false; }

    int index() const override { return 100000 + channel_index_; }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    // Check if the background thread is still running
    bool is_thread_alive() const
    {
        return frame_thread_.joinable();
    }
};

// A lightweight proxy that can respawn the inner ffmpeg_consumer on failure.
struct ffmpeg_consumer_proxy : public core::frame_consumer
{
    // Immutable constructor parameters for recreation
    const std::string        path_;
    const std::string        args_;
    const bool               realtime_;
    const common::bit_depth  depth_;

    // Last known initialization context for respawn
    core::video_format_desc  format_desc_{};
    std::optional<core::channel_info> channel_info_;
    int                      port_index_ = -1;

    // Inner consumer instance
    std::unique_ptr<ffmpeg_consumer> inner_;

    // Backoff state
    std::chrono::steady_clock::time_point last_failure_time_{};
    std::chrono::steady_clock::time_point last_send_time_{};
    int                                   consecutive_failures_ = 0;
    std::chrono::milliseconds             current_backoff_{0};
    
  public:
    ffmpeg_consumer_proxy(std::string path, std::string args, bool realtime, common::bit_depth depth)
        : path_(std::move(path))
        , args_(std::move(args))
        , realtime_(realtime)
        , depth_(depth)
    {
    }

    // frame_consumer
    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        format_desc_   = format_desc;
        channel_info_  = channel_info;
        port_index_    = port_index;

        inner_.reset();
        consecutive_failures_ = 0;
        current_backoff_      = std::chrono::milliseconds(0);
        
        // Create and initialize a fresh inner consumer; swallow exceptions to allow later retry
        try {
            inner_ = std::make_unique<ffmpeg_consumer>(path_, args_, realtime_, depth_);
            inner_->initialize(format_desc_, *channel_info_, port_index_);
        } catch (...) {
            // Don't log exception details - respawn will handle it
            CASPAR_LOG(info) << print() << L" Initial connection failed, will retry";
            inner_.reset();
            last_failure_time_ = std::chrono::steady_clock::now();
            consecutive_failures_++;
            current_backoff_ = std::chrono::milliseconds(1000);
        }
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        last_send_time_ = std::chrono::steady_clock::now();
        
        // If inner is missing (failed init), check backoff before attempting respawn
        if (!inner_) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_failure_time_);
            
            if (elapsed < current_backoff_) {
                // Still in backoff period; drop frame silently
                return caspar::make_ready_future(true);
            }

            // Backoff period elapsed; attempt respawn
            try {
                inner_ = std::make_unique<ffmpeg_consumer>(path_, args_, realtime_, depth_);
                inner_->initialize(format_desc_, *channel_info_, port_index_);
                
                // Success! Reset backoff state
                const auto prev_failures = consecutive_failures_;
                consecutive_failures_ = 0;
                current_backoff_      = std::chrono::milliseconds(0);
                if (prev_failures > 0) {
                    CASPAR_LOG(info) << print() << L" Reconnected after " << prev_failures << L" failure(s)";
                }
            } catch (...) {
                inner_.reset();
                last_failure_time_ = now;
                consecutive_failures_++;
                
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s (capped at 32s)
                current_backoff_ = std::chrono::milliseconds(
                    std::min(1000 * (1 << std::min(consecutive_failures_ - 1, 5)), 32000));
                
                CASPAR_LOG(warning) << print() << L" Connection failed (attempt " << consecutive_failures_ 
                                   << L"), retrying in " << current_backoff_.count() / 1000.0 << L"s";
                
                // Prevent removal by output: report success to keep proxy alive
                return caspar::make_ready_future(true);
            }
        }

        // Forward to inner and catch failures to trigger respawn on next call
        try {
            auto result = inner_->send(field, frame);
            auto proxy_ptr = this;
            
            // Wrap the future to catch exceptions when it's awaited
            return std::async(std::launch::deferred, [proxy_ptr, result = std::move(result)]() mutable {
                try {
                    bool success = result.get();
                    if (!success) {
                        throw std::runtime_error("Inner consumer returned false");
                    }
                    return true;
                } catch (...) {
                    proxy_ptr->inner_.reset();
                    proxy_ptr->last_failure_time_ = std::chrono::steady_clock::now();
                    proxy_ptr->consecutive_failures_++;
                    
                    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s (capped at 32s)
                    proxy_ptr->current_backoff_ = std::chrono::milliseconds(
                        std::min(1000 * (1 << std::min(proxy_ptr->consecutive_failures_ - 1, 5)), 32000));
                    
                    // Return true to prevent output.cpp from removing the proxy
                    return true;
                }
            });
        } catch (...) {
            // If inner_->send() itself throws (e.g., exception check at start), handle it here
            inner_.reset();
            last_failure_time_ = std::chrono::steady_clock::now();
            consecutive_failures_++;
            
            current_backoff_ = std::chrono::milliseconds(
                std::min(1000 * (1 << std::min(consecutive_failures_ - 1, 5)), 32000));
            
            if (consecutive_failures_ == 1) {
                CASPAR_LOG(error) << print() << L" Connection lost, reconnecting...";
            }
            
            return caspar::make_ready_future(true);
        }
    }

    std::future<bool> call(const std::vector<std::wstring>& params) override
    {
        // ffmpeg consumer does not implement call; safely no-op
        return caspar::make_ready_future(false);
    }

    std::wstring print() const override
    {
        if (inner_) return inner_->print();
        return L"ffmpeg[" + u16(path_) + L"]";
    }

    std::wstring name() const override { return L"ffmpeg"; }

    bool has_synchronization_clock() const override { return false; }

    int index() const override
    {
        // Mirror inner if present; otherwise derive from path checksum like inner does
        if (inner_) return inner_->index();
        boost::crc_16_type result;
        result.process_bytes(path_.data(), path_.length());
        return 100000 + result.checksum();
    }

    core::monitor::state state() const override
    {
        // Check if inner has failed even when not sending frames
        if (inner_) {
            try {
                // Try to detect if the inner consumer has a stored exception
                // by calling its state() which should be safe
                return inner_->state();
            } catch (...) {
                // If state() throws, the inner consumer has failed
                CASPAR_LOG(warning) << print() << L" Detected failure via state() check";
                // Can't modify mutable state from const method, but we can log
            }
        }
        
        core::monitor::state state;
        state["file/path"] = u8(path_);
        state["proxy/consecutive_failures"] = consecutive_failures_;
        state["proxy/backoff_ms"] = static_cast<int>(current_backoff_.count());
        return state;
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.size() < 2 || (!boost::iequals(params.at(0), L"STREAM") && !boost::iequals(params.at(0), L"FILE")))
        return core::frame_consumer::empty();

    auto                     path = u8(params.at(1));
    std::vector<std::string> args;
    bool                     respawn = false;
    for (auto n = 2; n < params.size(); ++n) {
        // Treat a bare RESPawn flag specially; otherwise collect as args
        if (boost::iequals(params[n], L"RESPAWN")) {
            respawn = true;
            continue;
        }
        args.emplace_back(u8(params[n]));
    }

    const auto realtime = boost::iequals(params.at(0), L"STREAM");

    if (respawn)
        return spl::make_shared<ffmpeg_consumer_proxy>(path, boost::join(args, " "), realtime, channel_info.depth);

    return spl::make_shared<ffmpeg_consumer>(path, boost::join(args, " "), realtime, channel_info.depth);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    const auto path     = u8(ptree.get<std::wstring>(L"path", L""));
    const auto args     = u8(ptree.get<std::wstring>(L"args", L""));
    const auto realtime = ptree.get(L"realtime", false);
    const auto respawn  = ptree.get(L"respawn", false);

    if (respawn)
        return spl::make_shared<ffmpeg_consumer_proxy>(path, args, realtime, channel_info.depth);

    return spl::make_shared<ffmpeg_consumer>(path, args, realtime, channel_info.depth);
}
}} // namespace caspar::ffmpeg
