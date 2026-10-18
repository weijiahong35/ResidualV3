from Baseline_native import install

if __name__ in ("__main__", "__mp_main__"):
    install()
if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    main()
