if __name__ in ('__main__', '__mp_main__'):
    from .target_prefix_bridge import install
    install()
    from .target_ragged_graph import install as install_ragged
    install_ragged()
    from .native_launch import install as install_repair
    install_repair()
    from .device_prefix import install as install_device_prefix
    install_device_prefix()
    from .short_latency import install as install_short_latency
    install_short_latency()
if __name__ == '__main__':
    from vllm.entrypoints.cli.main import main
    main()
