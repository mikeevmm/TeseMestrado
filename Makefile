cextensions: cextensions/*
	cd cextensions && make no_install=1 build
	EXTNAME="$$(find cextensions/build/lib.* -name "cextension.*.so")" && \
	cp $$EXTNAME ./ \
	&& mv "$$(basename $$EXTNAME)" cextension.so