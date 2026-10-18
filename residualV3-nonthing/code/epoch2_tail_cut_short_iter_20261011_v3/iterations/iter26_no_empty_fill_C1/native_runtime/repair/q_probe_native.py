if __name__ in ('__main__','__mp_main__'):
    from .native_launch import install
    install()
    from .trained_capture import install_capture
    install_capture()
if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
